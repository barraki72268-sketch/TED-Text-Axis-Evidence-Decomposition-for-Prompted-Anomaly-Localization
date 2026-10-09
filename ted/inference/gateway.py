"""Route image inspection to separately isolated, artifact-pinned host workers."""
import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import re
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
import httpx


def load_registry(path):
    rows = json.loads(Path(path).read_text(encoding='utf-8'))['models']
    registry = {}
    for row in rows:
        name, url, sha = row['id'], row['url'], row['artifact_sha256']
        parsed = urlsplit(url)
        if (not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,79}', name) or name in registry or
                not re.fullmatch(r'[a-f0-9]{64}', sha) or parsed.scheme != 'http' or
                not parsed.hostname or parsed.username or parsed.password or
                parsed.path not in {'', '/'} or parsed.query or parsed.fragment):
            raise ValueError('Invalid or duplicate worker registration')
        registry[name] = dict(row, url=url.rstrip('/'))
    if not registry:
        raise ValueError('Register at least one verified worker')
    return registry


def create_app(registry=None, *, transport=None, max_body_bytes=20_000_000):
    registry = registry if registry is not None else load_registry(os.environ['TED_MODEL_REGISTRY'])
    locks = {name: asyncio.Lock() for name in registry}

    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(transport=transport, trust_env=False, follow_redirects=False,
                                     timeout=httpx.Timeout(120, connect=3)) as client:
            app.state.client = client
            yield

    app = FastAPI(title='TED model inspection', lifespan=lifespan)

    def selected(name):
        if name not in registry:
            raise HTTPException(404, 'Unknown registered model')
        return registry[name]

    async def worker_info(name):
        entry = selected(name)
        try:
            response = await app.state.client.get(entry['url'] + '/model-info', timeout=5)
            response.raise_for_status()
            info = response.json()
            if info.get('artifact_sha256') != entry['artifact_sha256']:
                raise HTTPException(503, 'Worker artifact differs from the registered release')
            return info
        except (httpx.HTTPError, ValueError, AttributeError):
            raise HTTPException(503, 'Model worker unavailable')

    async def inventory_entry(name):
        entry = selected(name)
        result = dict(id=name, label=entry['label'], scope=entry['scope'],
                      artifact_sha256=entry['artifact_sha256'])
        try:
            result.update(ready=True, info=await worker_info(name))
        except HTTPException:
            result.update(ready=False, info=None)
        return result

    @app.get('/health')
    async def health():
        return dict(status='alive')

    @app.get('/models')
    async def models():
        return dict(models=await asyncio.gather(*(inventory_entry(name) for name in registry)))

    @app.get('/ready')
    async def ready():
        inventory = await models()
        if not all(row['ready'] for row in inventory['models']):
            raise HTTPException(503, 'One or more registered workers are unavailable')
        return dict(status='ready', models=len(registry))

    @app.get('/model-info')
    async def model_info(model: str):
        return await worker_info(model)

    @app.post('/predict')
    async def predict(request: Request, model: str, category: str | None = None):
        entry = selected(model)
        content_type = request.headers.get('content-type', '').split(';')[0].strip().lower()
        if content_type not in {'image/png', 'image/jpeg'}:
            raise HTTPException(415, 'Send raw PNG/JPEG bytes with an image Content-Type')
        lock = locks[model]
        if lock.locked():
            raise HTTPException(503, 'Inference busy; retry later', headers={'Retry-After': '1'})
        async with lock:
            await worker_info(model)
            body = bytearray()
            async for chunk in request.stream():
                if len(body) + len(chunk) > max_body_bytes:
                    raise HTTPException(413, 'Image byte limit exceeded')
                body.extend(chunk)
            try:
                response = await app.state.client.post(entry['url'] + '/predict',
                    params={'category': category} if category is not None else {},
                    content=bytes(body), headers={'Content-Type': content_type})
                if response.status_code == 200:
                    result = response.json()
                    if result.get('artifact_sha256') != entry['artifact_sha256']:
                        raise HTTPException(502, 'Prediction artifact differs from the registered release')
                elif response.status_code >= 500 and response.status_code != 503:
                    raise HTTPException(502, 'Model worker failed')
                return Response(content=response.content, status_code=response.status_code,
                    media_type='application/json', headers=dict(
                        {'X-TED-Model': model}, **({'Retry-After': '1'} if response.status_code == 503 else {})))
            except httpx.TimeoutException:
                raise HTTPException(504, 'Model worker timed out')
            except (httpx.HTTPError, ValueError, AttributeError):
                raise HTTPException(502, 'Invalid model worker response')

    @app.get('/', response_class=HTMLResponse)
    async def index():
        return Path(__file__).with_name('gateway.html').read_text(encoding='utf-8')

    return app
