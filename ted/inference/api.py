"""Private, single-worker HTTP service. Send JPEG/PNG bytes to POST /predict.

Run behind an authenticated ingress before any external exposure. This module
does not download weights, fit calibration, accept model paths from clients,
or persist submitted images. The host still needs a trusted research checkout.
"""
import base64
from contextlib import asynccontextmanager
import io
import json
import logging
import math
import os
import threading
import time
import uuid
import warnings

from fastapi import FastAPI, HTTPException, Request
import numpy as np
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool


LOG = logging.getLogger("uvicorn.error")


def engine_from_env():
    from .engine import FAPromptEngine
    return FAPromptEngine(artifact_path=os.environ["TED_ARTIFACT"],
                         checkpoint_path=os.environ["TED_CHECKPOINT"],
                         research_root=os.environ["TED_RESEARCH_ROOT"],
                         device=os.environ.get("TED_DEVICE", "cpu"))


def encode_prediction(engine, body, max_pixels):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(body)) as opened:
                if opened.format not in {"PNG", "JPEG"}:
                    raise HTTPException(415, "Only PNG and JPEG are supported")
                if opened.width * opened.height > max_pixels:
                    raise HTTPException(413, "Image pixel limit exceeded")
                image = opened.convert("RGB")
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise HTTPException(413, "Image pixel limit exceeded")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(400, "Invalid or truncated image")
    output = engine.predict(image)
    host = np.asarray(output["host_map"], dtype=np.float32)
    cted = np.asarray(output["cted_map"], dtype=np.float32)
    if (host.ndim != 3 or host.shape[0] != 1 or host.shape != cted.shape
            or not np.isfinite(host).all() or not np.isfinite(cted).all()
            or not math.isfinite(output["image_score"])):
        raise RuntimeError("Invalid model output")
    raw = io.BytesIO()
    np.savez_compressed(raw, host=host, cted=cted)
    low, high = np.percentile(np.concatenate([host.ravel(), cted.ravel()]), [2, 99.5])
    previews = {}
    for name, values in (("host", host), ("cted", cted)):
        pixels = np.clip((values[0] - low) / max(float(high - low), 1e-8), 0, 1)
        buffer = io.BytesIO()
        Image.fromarray((pixels * 255).astype("uint8")).save(buffer, format="PNG")
        previews[name] = base64.b64encode(buffer.getvalue()).decode("ascii")
    return dict(image_score=output["image_score"],
                image_score_policy="unchanged official host score, not a probability",
                artifact_sha256=output["artifact_sha256"],
                map_shape=list(cted.shape), maps_npz_base64=base64.b64encode(raw.getvalue()).decode("ascii"),
                previews_png_base64=previews,
                display=dict(percentiles=[2, 99.5], shared_range=[float(low), float(high)]),
                timing_ms=output["timing_ms"])


def create_app(engine_factory=None, *, max_body_bytes=20_000_000, max_pixels=25_000_000):
    factory = engine_factory or engine_from_env

    @asynccontextmanager
    async def lifespan(app):
        app.state.engine = await run_in_threadpool(factory)
        try:
            yield
        finally:
            app.state.engine = None

    app = FastAPI(title="TED private inspection service", version="0.1.0", lifespan=lifespan)
    app.state.engine = None
    admission = threading.Lock()

    @app.get("/health")
    def health():
        return {"status": "alive"}

    @app.get("/ready")
    def ready():
        if app.state.engine is None:
            raise HTTPException(503, "Model not ready")
        return {"status": "ready"}

    @app.get("/model-info")
    def model_info():
        if app.state.engine is None:
            raise HTTPException(503, "Model not ready")
        return app.state.engine.info()

    @app.post("/predict")
    async def predict(request: Request):
        request_id = uuid.uuid4().hex
        started = time.perf_counter()
        status = 500
        acquired = False
        try:
            if app.state.engine is None:
                raise HTTPException(503, "Model not ready")
            if request.headers.get("content-type", "").split(";")[0].strip().lower() not in {"image/png", "image/jpeg"}:
                raise HTTPException(415, "Send raw PNG/JPEG bytes with an image Content-Type")
            acquired = admission.acquire(blocking=False)
            if not acquired:
                raise HTTPException(503, "Inference busy; retry later", headers={"Retry-After": "1"})
            body = bytearray()
            async for chunk in request.stream():
                if len(body) + len(chunk) > max_body_bytes:
                    raise HTTPException(413, "Image byte limit exceeded")
                body.extend(chunk)
            result = await run_in_threadpool(encode_prediction, app.state.engine, body, max_pixels)
            result["request_id"] = request_id
            # Includes decode/NPZ/PNG encoding, excludes HTTP transport/JSON send.
            result["server_processing_ms"] = (time.perf_counter() - started) * 1000
            status = 200
            return result
        except HTTPException as exc:
            status = exc.status_code
            raise
        except Exception:
            LOG.exception("TED inference failed; request_id=%s", request_id)
            raise HTTPException(500, "Inference failed", headers={"X-Request-ID": request_id})
        finally:
            if acquired:
                admission.release()
            LOG.info(json.dumps(dict(event="prediction", request_id=request_id, status=status,
                                     duration_ms=round((time.perf_counter() - started) * 1000, 3))))

    return app
