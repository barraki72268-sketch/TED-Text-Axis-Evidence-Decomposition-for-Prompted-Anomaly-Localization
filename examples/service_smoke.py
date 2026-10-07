"""Start a loopback-only real server, check HTTP outputs, and shut it down.

Set TED_ARTIFACT/CHECKPOINT/RESEARCH_ROOT/DEVICE and the host cache environment.
Run GPU versions inside your scheduler allocation. This is not a load benchmark.
"""
import argparse
import base64
import io
import json
from pathlib import Path
import subprocess
import sys
import time

import httpx
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--reference-maps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    address = f"http://127.0.0.1:{args.port}"
    with (args.output / "server.log").open("w") as log:
        process = subprocess.Popen([sys.executable, "-m", "uvicorn", "ted.inference.api:create_app",
                                    "--factory", "--host", "127.0.0.1", "--port", str(args.port),
                                    "--workers", "1"], stdout=log, stderr=subprocess.STDOUT)
        try:
            with httpx.Client(base_url=address, timeout=120, trust_env=False) as client:
                deadline = time.monotonic() + 120
                while True:
                    if process.poll() is not None:
                        raise RuntimeError("Server exited during startup; inspect server.log")
                    try:
                        if client.get("/ready").status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError("Startup timed out")
                    time.sleep(0.25)
                info = client.get("/model-info")
                info.raise_for_status()
                reference = np.load(args.reference_maps, allow_pickle=False)
                observations = []
                for _ in range(3):
                    started = time.perf_counter()
                    response = client.post("/predict", content=args.image.read_bytes(),
                                           headers={"Content-Type": "image/png"})
                    response.raise_for_status()
                    elapsed = (time.perf_counter() - started) * 1000
                    result = response.json()
                    with np.load(io.BytesIO(base64.b64decode(result["maps_npz_base64"])), allow_pickle=False) as maps:
                        errors = {}
                        for name in ("host", "cted"):
                            errors[name] = float(np.max(np.abs(maps[name] - reference[name])))
                            np.testing.assert_allclose(maps[name], reference[name], atol=2e-5, rtol=2e-5)
                    observations.append(dict(request_id=result["request_id"], image_score=result["image_score"],
                                             client_roundtrip_ms=elapsed, timing_ms=result["timing_ms"],
                                             server_processing_ms=result["server_processing_ms"],
                                             max_abs_error_vs_cli=errors))
                invalid = client.post("/predict", content=b"invalid", headers={"Content-Type": "image/png"})
                assert invalid.status_code == 400
                report = dict(model=info.json(), requests=observations, invalid_input_status=invalid.status_code,
                              scope="single-image FP32 HTTP smoke; not dataset reproduction or a latency benchmark")
                (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(report, indent=2))
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


if __name__ == "__main__":
    main()
