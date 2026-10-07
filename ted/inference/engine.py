"""Reusable single-image engine backed by a trusted local research checkout.

Initialize once at service startup. Each predict call performs inference only.
This bridge deliberately does not claim a self-contained upstream host port.
"""
import importlib.util
from pathlib import Path
import threading
from types import SimpleNamespace
import time

import torch

from .faprompt import BranchCalibratedReadout, sha256


class FAPromptEngine:
    def __init__(self, *, artifact_path, checkpoint_path, research_root, device="cpu"):
        started = time.perf_counter()
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
        script = Path(research_root) / "neurips2026/scripts/official_parallel_test_faprompt.py"
        if sha256(checkpoint_path) != artifact["checkpoint_sha256"] or sha256(script) != artifact["research_sha256"]:
            raise ValueError("Checkpoint/research evaluator does not match artifact")
        self.artifact_sha256 = sha256(artifact_path)
        self.readout = BranchCalibratedReadout(artifact, device)
        self.device = self.readout.device
        self.cfg, self.options = artifact["interface"], artifact["inference"]
        spec = importlib.util.spec_from_file_location("ted_trusted_research_host", script)
        self.host = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.host)
        self.model, self.prompts, _ = self.host.build_model(self.device, str(checkpoint_path),
            self.cfg["model_name"], self.cfg["prompt_load_mode"], self.cfg["dpam_layer"],
            self.options["depth"], self.options["n_ctx"], self.options["t_n_ctx"])
        self.transform, _ = self.host.get_transform(SimpleNamespace(image_size=self.cfg["image_size"]))
        with torch.inference_mode():
            self.pair = self.host.learned_text_pair(self.model, self.prompts)
            axis = torch.nn.functional.normalize(self.pair[1] - self.pair[0], dim=0).cpu()
            torch.testing.assert_close(axis, artifact["axis"], atol=1e-5, rtol=1e-5)
        self._lock = threading.Lock()
        self._sync()
        self.model_load_ms = (time.perf_counter() - started) * 1000

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def info(self):
        return dict(host="FAPrompt", variant="C-TED branch_calibrated",
                    artifact_sha256=self.artifact_sha256, device=str(self.device),
                    input_size=self.cfg["image_size"], model_load_ms=self.model_load_ms,
                    deployment_status="experimental local research bridge")

    @torch.inference_mode()
    def predict(self, image):
        if image.width * image.height > 25_000_000:
            raise ValueError("Image exceeds 25 million pixels")
        # The host may maintain internal mutable state. Do not overlap requests.
        # A future API should map busy to 503, not create an unbounded queue.
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Inference engine is busy")
        try:
            started = time.perf_counter()
            tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
            self._sync()
            preprocessing_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            output = self.host.compute_faprompt_outputs(self.model, self.prompts, tensor,
                self.cfg["features_list"], self.cfg["image_size"], self.options["sigma"],
                self.cfg["dap_token_mode"], self.cfg["dpam_layer"], self.pair)
            self._sync()
            host_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            corrected = self.readout(output["patch_tokens"], output["token_score"],
                                    output["branch1_token_score"], output["branch2_token_score"])
            maps = self.host.upsample_tokens(corrected, self.cfg["image_size"], self.options["sigma"])
            self._sync()
            cted_ms = (time.perf_counter() - started) * 1000
            return dict(host_map=output["anomaly_map"], cted_map=maps,
                        image_score=float(output["official_image_score"][0]),
                        timing_ms=dict(preprocessing=preprocessing_ms, host_including_postprocess=host_ms,
                                       cted_including_postprocess=cted_ms),
                        artifact_sha256=self.artifact_sha256)
        finally:
            self._lock.release()
