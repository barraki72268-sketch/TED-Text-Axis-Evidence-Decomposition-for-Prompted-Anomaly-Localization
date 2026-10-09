import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import threading
import unittest

from fastapi.testclient import TestClient
import httpx

from ted.inference.gateway import create_app, load_registry


class GatewayTests(unittest.TestCase):
    def test_busy_worker_rejects_duplicate_without_blocking_other_models(self):
        entered, release = threading.Event(), threading.Event()
        async def handler(request):
            sha = ('a' if request.url.host == 'aa' else 'b') * 64
            if request.url.path == '/predict' and request.url.host == 'aa':
                entered.set()
                await asyncio.to_thread(release.wait, 5)
            return httpx.Response(200, json=dict(artifact_sha256=sha))
        with self.fixture(handler) as client, ThreadPoolExecutor() as pool:
            def predict(model):
                return client.post('/predict?model=' + model, content=b'image', headers={'Content-Type': 'image/png'})
            future = pool.submit(predict, 'aa')
            try:
                self.assertTrue(entered.wait(3))
                busy = predict('aa')
                self.assertEqual(busy.status_code, 503)
                self.assertEqual(busy.headers['Retry-After'], '1')
                self.assertEqual(predict('fap').status_code, 200)
            finally:
                release.set()
            self.assertEqual(future.result(timeout=3).status_code, 200)

    def fixture(self, handler, limit=20_000_000):
        registry = {n: dict(id=n, label=n, scope='test-only', url='http://' + n,
                          artifact_sha256=sha * 64) for n, sha in [('aa', 'a'), ('fap', 'b')]}
        return TestClient(create_app(registry, transport=httpx.MockTransport(handler), max_body_bytes=limit))

    def test_selects_worker_and_preserves_exact_maps_scores_and_category(self):
        seen = []
        def handler(request):
            sha = ('a' if request.url.host == 'aa' else 'b') * 64
            if request.url.path == '/model-info':
                return httpx.Response(200, json=dict(artifact_sha256=sha, categories=['01']))
            seen.append((request.url.host, request.url.params.get('category'), request.content))
            return httpx.Response(200, content=json.dumps(dict(artifact_sha256=sha,
                maps_npz_base64='unchanged-payload', image_score=.321)).encode())
        with self.fixture(handler) as client:
            self.assertEqual(client.get('/ready').json()['models'], 2)
            self.assertEqual(len(client.get('/models').json()['models']), 2)
            for model, sha in [('aa', 'a'), ('fap', 'b')]:
                response = client.post('/predict?model=' + model + '&category=01',
                                       content=b'image-fixture', headers={'Content-Type': 'image/png'})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers['X-TED-Model'], model)
                self.assertEqual(response.json(), dict(artifact_sha256=sha * 64,
                    maps_npz_base64='unchanged-payload', image_score=.321))
            self.assertEqual(seen, [('aa', '01', b'image-fixture'), ('fap', '01', b'image-fixture')])
            self.assertIn('Model release', client.get('/').text)

    def test_rejects_wrong_artifact_before_inference_and_after_prediction(self):
        for wrong_info in [True, False]:
            seen = []
            def handler(request):
                seen.append(request.url.path)
                sha = 'c' if wrong_info or request.url.path == '/predict' else 'a'
                return httpx.Response(200, json=dict(artifact_sha256=sha * 64))
            with self.fixture(handler) as client:
                response = client.post('/predict?model=aa', content=b'image', headers={'Content-Type': 'image/png'})
                self.assertEqual(response.status_code, 503 if wrong_info else 502)
            self.assertEqual('/predict' in seen, not wrong_info)

    def test_input_limits_unknown_models_and_worker_errors(self):
        def handler(request):
            if request.url.path == '/model-info':
                return httpx.Response(200, json=dict(artifact_sha256='a' * 64))
            return httpx.Response(422, json=dict(detail='Unknown target category'))
        with self.fixture(handler, limit=3) as client:
            self.assertEqual(client.post('/predict?model=missing', content=b'x').status_code, 404)
            self.assertEqual(client.post('/predict?model=aa', content=b'x').status_code, 415)
            self.assertEqual(client.post('/predict?model=aa', content=b'abcd', headers={'Content-Type': 'image/png'}).status_code, 413)
            self.assertEqual(client.post('/predict?model=aa', content=b'ab', headers={'Content-Type': 'image/png'}).status_code, 422)

    def test_unavailable_worker_is_sanitized_and_registry_is_fixed(self):
        def handler(request):
            raise httpx.ConnectError('private backend detail', request=request)
        with self.fixture(handler) as client:
            self.assertEqual(client.get('/ready').status_code, 503)
            self.assertFalse(client.get('/models').json()['models'][0]['ready'])
            response = client.get('/model-info?model=aa')
            self.assertNotIn('private backend detail', response.text)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'registry.json'
            row = dict(id='aa', label='AA', scope='test-only', artifact_sha256='a' * 64, url='http://localhost:8000')
            for invalid in ['http://user:password@localhost', 'http://localhost/path', 'https://localhost']:
                path.write_text(json.dumps({'models': [dict(row, url=invalid)]}))
                with self.assertRaises(ValueError):
                    load_registry(path)
            path.write_text(json.dumps({'models': [row, row]}))
            with self.assertRaises(ValueError):
                load_registry(path)
