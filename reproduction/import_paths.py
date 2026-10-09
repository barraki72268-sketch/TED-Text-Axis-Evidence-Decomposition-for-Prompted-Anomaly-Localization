"""Keep replay-wrapper modules out of upstream top-level import resolution."""
from pathlib import Path


def isolated_paths(paths, script, cwd, runner_directory):
    runner = Path(runner_directory).resolve()
    front = [str(Path(script).resolve().parent), str(Path(cwd).resolve())]
    remaining = [p for p in paths if Path(p or '.').resolve() != runner and p not in front]
    return front + remaining
