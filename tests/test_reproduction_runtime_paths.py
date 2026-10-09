import json
import os
from pathlib import Path
import tempfile
import unittest

from reproduction.backbone_download import required_backbone_assets
from reproduction.runtime import link_runtime_asset, relocate_computed_dataset_roots


ROOT = Path(__file__).resolve().parents[1] / "reproduction"


class RuntimePathTests(unittest.TestCase):
    def test_computed_dataset_roots_bind_full_prepared_metadata_without_changing_settings(self):
        text = ('ROOT = Path("/original")\n'
                'OFFICIAL_VISA_ROOT = ROOT / "neurips2026" / "data" / "VisA_pytorch_official" / "1cls"\n'
                'BTAD_ROOT = ROOT / "neurips2026" / "data" / "BTAD_official"\n'
                'batch_size = 4\nsigma = 8\n')
        bound = relocate_computed_dataset_roots(text, {"visa": Path("/verified/visa"), "btad": Path("/verified/btad")})
        namespace = {"Path": Path}
        exec(bound, namespace)
        self.assertEqual(namespace['OFFICIAL_VISA_ROOT'], Path('/verified/visa'))
        self.assertEqual(namespace['BTAD_ROOT'], Path('/verified/btad'))
        self.assertIn('batch_size = 4\nsigma = 8\n', bound)
        with self.assertRaisesRegex(ValueError, "computed dataset root differs"):
            relocate_computed_dataset_roots(text.replace('"BTAD_official"', '"different_protocol"'), {"btad": Path('/verified/btad')})

    def test_aa_bootstrap_does_not_replace_selected_backbone(self):
        catalog = json.loads((ROOT / "backbones.json").read_text())
        for binding in catalog["bindings"]:
            assets = required_backbone_assets(catalog, binding["recipe"])
            self.assertEqual(assets[0]["sha256"], binding["sha256"])
            if binding["host"] == "AA-CLIP":
                self.assertIn("openai_vit_l14_336", {a["id"] for a in assets})
            else:
                self.assertEqual(len(assets), 1)

    @unittest.skipIf(os.name == "nt", "Linux research runtime uses POSIX symlinks; exercised by Linux CI")
    def test_fap_computed_checkpoint_path_and_conflict_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, checkpoint = root / "source", root / "checkpoint-object"
            checkpoint.write_bytes(b"verified-checkpoint")
            relative = Path("neurips2026/FAPrompt/checkpoints/trained_on_mvtecad/epoch_15.pth")
            link_runtime_asset(source, relative, checkpoint)
            # This is the FAP evaluator's Path(__file__).parents[2]-based lookup.
            evaluator = source / "neurips2026/scripts/official_parallel_test_faprompt.py"
            computed = evaluator.parents[2] / relative
            self.assertEqual(computed.read_bytes(), checkpoint.read_bytes())
            other = root / "different-object"
            other.write_bytes(b"different")
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                link_runtime_asset(source, relative, other)
            self.assertEqual(computed.read_bytes(), b"verified-checkpoint")
            with self.assertRaisesRegex(ValueError, "within"):
                link_runtime_asset(source, Path("../escape"), checkpoint)
