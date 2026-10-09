import tempfile
from pathlib import Path
import unittest

from ted.inference.adaptclip_engine import load_relocated_module


class RelocationTests(unittest.TestCase):
    def test_only_path_binding_changes_and_original_bytes_stay_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            script = Path(temp) / 'host.py'
            original = b'from pathlib import Path\nROOT = Path("/old/source")\ndef numeric(x):\n    return x * 3.25 + 1.0\n'
            script.write_bytes(original)
            module = load_relocated_module('fixture', script, Path(temp) / 'relocated')
            self.assertEqual(module.ROOT, Path(temp) / 'relocated')
            self.assertEqual(module.numeric(2), 7.5)
            self.assertEqual(script.read_bytes(), original)

    def test_ambiguous_or_computed_root_is_rejected_before_module_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            script = Path(temp) / 'host.py'
            for source in ['ROOT=Path("a")\nROOT=Path("b")', 'ROOT=Path(root_from_env())', 'x=1']:
                script.write_text(source+'\nraise AssertionError("must not execute")\n')
                with self.assertRaises(ValueError):
                    load_relocated_module('fixture', script, Path(temp))
