"""Record fresh Figure 4 RNG state before invoking the unchanged collector."""
import importlib.metadata
import json
from pathlib import Path
import random
import runpy
import sys


def main():
    # This script is launched by absolute path, outside the package working dir.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from reproduction.axis_run import require_live_gpu_allocation
    allocation = require_live_gpu_allocation()
    collector, state_record, *arguments = sys.argv[1:]
    namespace = runpy.run_path(collector, run_name='figure4_prepared_collector')
    import numpy as np
    import torch
    state = np.random.get_state()
    report = {
        'scope': 'Observed fresh RNG after imports and before model construction; no added seed or claim of historical RNG provenance',
        'allocation': allocation, 'python_random': random.getstate(),
        'numpy_random': [state[0], state[1].tolist(), *state[2:]],
        'torch_cpu_rng_hex': bytes(torch.get_rng_state().tolist()).hex(),
        'torch_cuda_rng_hex': [bytes(value.tolist()).hex() for value in torch.cuda.get_rng_state_all()],
        'python': sys.version,
        'versions': {name: importlib.metadata.version(name) for name in
                     ['numpy', 'scikit-learn', 'torch', 'torchvision', 'matplotlib', 'Pillow']},
    }
    with Path(state_record).open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    require_live_gpu_allocation()
    sys.argv = [collector, *arguments]
    namespace['main']()


if __name__ == '__main__':
    main()
