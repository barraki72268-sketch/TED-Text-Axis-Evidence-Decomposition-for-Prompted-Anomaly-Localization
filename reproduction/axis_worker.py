"""Internal collector worker: retain initial RNG state without selecting a seed."""
import importlib.metadata
import json
import os
from pathlib import Path
import random
import runpy
import sys


def main():
    collector, state_record, *arguments = sys.argv[1:]
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise RuntimeError('Internal Figure 3 worker requires a GPU Slurm allocation')
    namespace = runpy.run_path(collector, run_name='figure3_prepared_collector')
    import numpy as np
    import torch
    state = np.random.get_state()
    report = {
        'scope': 'Observed fresh initial RNG states after collector import and before model construction; not historical RNG provenance',
        'python_random': random.getstate(),
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
    sys.argv = [collector, *arguments]
    namespace['main']()


if __name__ == '__main__':
    main()
