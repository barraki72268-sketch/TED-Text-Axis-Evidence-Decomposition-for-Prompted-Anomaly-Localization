from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from reproduction.import_paths import isolated_paths


class EvaluatorImportTests(unittest.TestCase):
    def test_upstream_metrics_wins_over_wrapper_metrics_in_fresh_process(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            runner, upstream, source = [base / n for n in ['runner', 'upstream', 'source']]
            for p in [runner, upstream, source]:
                p.mkdir()
            (runner / 'metrics.py').write_text("raise RuntimeError('wrong metrics module')")
            (upstream / 'metrics.py').write_text('AUPR = 123')
            code = ('import sys; from reproduction.import_paths import isolated_paths; '
                    f'sys.path[:] = isolated_paths([str({str(runner)!r}),str({str(upstream)!r})]+sys.path, '
                    f'{str(source / "eval.py")!r}, {str(source)!r}, {str(runner)!r}); '
                    'from metrics import AUPR; assert AUPR == 123')
            result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_only_runner_directory_is_removed_and_research_order_preserved(self):
        base = Path.cwd()
        runner, source, helper = base / 'runner', base / 'source', base / 'source/tools'
        result = isolated_paths([str(runner), str(helper), str(source), '/other'], helper / 'eval.py', source, runner)
        self.assertEqual(result[:2], [str(helper.resolve()), str(source.resolve())])
        self.assertNotIn(str(runner), result)
        self.assertIn('/other', result)
