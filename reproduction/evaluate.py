"""Isolated evaluator process; compatibility/capture matches the research replay."""
import functools
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch


def main():
    if not hasattr(np, "trapezoid"):
        np.trapezoid = np.trapz
    plan = json.loads(Path(sys.argv[1]).read_text())
    script, argv = Path(plan["evaluator"]), plan["argv"]
    output = Path(argv[argv.index("--save_dir") + 1])
    capture = output / "artifacts"
    capture.mkdir(parents=True, exist_ok=True)
    (output / "environment.json").write_text(json.dumps({
        "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }, indent=2))
    sys.path.insert(0, str(script.parent))
    sys.path.insert(1, plan["cwd"])
    spec = importlib.util.spec_from_file_location("ted_replay_evaluator", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def frozen(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value.copy())
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {key: frozen(item) for key, item in value.items()}
        if isinstance(value, list):
            return [frozen(item) for item in value]
        if isinstance(value, tuple):
            return tuple(frozen(item) for item in value)
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        raise TypeError(f"Unexpected capture value: {type(value).__name__}")

    index = []
    def wrapper(function, name):
        @functools.wraps(function)
        def call(*args, **kwargs):
            result = function(*args, **kwargs)
            if result is not None:
                path = capture / f"{len(index):03d}-{name}.pt"
                torch.save(frozen(result), path)
                index.append({"function": name, "file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
                (capture / "index.json").write_text(json.dumps(index, indent=2))
            return result
        return call
    names = ["get_or_collect_source_banks", "get_or_collect_source_patch_banks", "get_or_collect_banks",
             "train_prescore_calibrators", "train_prescore_map_calibrators", "train_layer_calibrators",
             "train_source_calibrators", "train_rawclip_source_calibrators", "train_subspace_host_residual_calibrator",
             "train_vl_subspace_calibrator", "train_vl_rankr_readout_calibrator", "build_vl_rankr_density_calibrator"]
    for name in names:
        if callable(getattr(module, name, None)):
            setattr(module, name, wrapper(getattr(module, name), name))
    sys.argv = [str(script), *argv]
    module.main()


if __name__ == "__main__":
    main()
