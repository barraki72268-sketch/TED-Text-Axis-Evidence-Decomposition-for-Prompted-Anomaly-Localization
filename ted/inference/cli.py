"""Experimental local research bridge (trusted files only).

Export: python -m ted.inference.cli export --help
Infer:  python -m ted.inference.cli infer --help
The original research checkout and its dependencies are still required for the
host. This bridge is not a standalone deployment package or public API server.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .faprompt import export_artifact, sha256
from .engine import FAPromptEngine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="Offline source-only calibration; does not load host")
    export.add_argument("--bank", type=Path, required=True)
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--research-root", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--alpha", type=float, required=True)
    export.add_argument("--seed", type=int, default=0)
    export.add_argument("--epochs", type=int, default=30)
    infer = sub.add_parser("infer", help="Single-image inference; never fits a calibrator")
    infer.add_argument("--artifact", type=Path, required=True)
    infer.add_argument("--checkpoint", type=Path, required=True)
    infer.add_argument("--research-root", type=Path, required=True)
    infer.add_argument("--image", type=Path, required=True)
    infer.add_argument("--output-dir", type=Path, required=True)
    infer.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.command == "export":
        if args.output.exists():
            raise FileExistsError("Choose a new artifact path; existing artifacts are not overwritten")
        script = args.research_root / "neurips2026/scripts/official_parallel_test_faprompt.py"
        banks = torch.load(args.bank, map_location="cpu", weights_only=True)
        artifact = export_artifact(banks, checkpoint_sha256=sha256(args.checkpoint),
                                   research_sha256=sha256(script), alpha=args.alpha,
                                   seed=args.seed, epochs=args.epochs)
        artifact["source_bank_sha256"] = sha256(args.bank)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(artifact, args.output)
        print(json.dumps({"artifact": str(args.output), "sha256": sha256(args.output)}))
        return
    if args.output_dir.exists():
        raise FileExistsError("Choose a new output directory")
    # Validate input before loading the heavyweight model. No target masks.
    if args.image.stat().st_size > 20_000_000:
        raise ValueError("Image file exceeds 20 MB")
    with Image.open(args.image) as opened:
        if opened.width * opened.height > 25_000_000:
            raise ValueError("Image exceeds 25 million pixels")
        image = opened.convert("RGB")
    engine = FAPromptEngine(artifact_path=args.artifact, checkpoint_path=args.checkpoint,
                           research_root=args.research_root, device=args.device)
    output = engine.predict(image)
    maps = output["cted_map"]
    args.output_dir.mkdir(parents=True)
    np.savez_compressed(args.output_dir / "maps.npz", host=output["host_map"], cted=maps)
    # Raw maps above remain unnormalized; previews use a paired display range.
    lo, hi = np.percentile(np.concatenate([output["host_map"].ravel(), maps.ravel()]), [2, 99.5])
    for name, values in (("host", output["host_map"]), ("cted", maps)):
        pixels = np.clip((values[0] - lo) / max(hi - lo, 1e-8), 0, 1)
        Image.fromarray((pixels * 255).astype("uint8")).save(args.output_dir / f"{name}_preview.png")
    report = dict(artifact_sha256=sha256(args.artifact), device=args.device,
                  image_sha256=sha256(args.image), image_score=output["image_score"],
                  image_score_policy="unchanged official FAPrompt score; not calibrated probability",
                  model_load_ms=engine.model_load_ms, cold_single_image_timing_ms=output["timing_ms"],
                  timing_note="No warm-up; not a latency benchmark. Excludes output serialization.",
                  display_percentiles=[2, 99.5], display_range=[float(lo), float(hi)])
    (args.output_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
