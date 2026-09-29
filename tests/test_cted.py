import ast
import hashlib
import json
from pathlib import Path
import unittest
import torch
from examples.cted_synthetic import run
from ted.cted import aaclip, adaclip, faprompt


class CTEDTests(unittest.TestCase):
    def test_source_function_integrity(self):
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root / "docs/cted_source_manifest.json").read_text())
        for host, record in manifest.items():
            source = (root / f"ted/cted/{host}.py").read_text()
            functions = {n.name: ast.get_source_segment(source, n) for n in ast.parse(source).body
                         if isinstance(n, ast.FunctionDef)}
            for name, digest in record["functions"].items():
                self.assertEqual(hashlib.sha256(functions[name].encode()).hexdigest(), digest)

    def test_train_inference_bounds_and_determinism(self):
        for host in ["aaclip", "adaclip", "faprompt"]:
            for seed in [0, 1, 2]:
                with self.subTest(host=host, seed=seed):
                    cal, score = run(host, seed)
                    other, score2 = run(host, seed)
                    self.assertTrue(torch.isfinite(score).all())
                    self.assertTrue(0 <= cal["eta"] <= 0.200001)
                    self.assertTrue(0 <= cal["readout_gamma"] <= 0.200001)
                    basis = cal["basis"]
                    torch.testing.assert_close(basis.T @ basis, torch.eye(basis.shape[1]), atol=1e-5, rtol=1e-5)
                    torch.testing.assert_close(score, score2, atol=0, rtol=0)
                    self.assertEqual(cal["transport_direction"], other["transport_direction"])

    def test_zero_residual_preserves_host(self):
        tokens = torch.randn(1, 16, 12)
        axis = torch.eye(12)[:, 0]
        basis = torch.eye(12)[:, :4]
        host = tokens @ axis
        for module in [adaclip, faprompt]:
            score = module.subspace_host_residual_token_score(host, tokens, basis, [1., 0., 0.], 0.)
            torch.testing.assert_close(score, host)
        score = aaclip.subspace_host_residual_map_from_tokens(
            tokens, tokens, basis, [1., 0., 0.], axis, 0., 4, "host_residual")
        torch.testing.assert_close(score, host.view(1, 4, 4))

    def test_adaclip_logit_center_preserved(self):
        logits = torch.randn(2, 2, 4, 4)
        delta = torch.randn(2, 4, 4)
        updated = adaclip.add_to_logit_difference(logits, delta)
        torch.testing.assert_close(updated.mean(1), logits.mean(1))
        torch.testing.assert_close(updated[:, 1] - updated[:, 0], logits[:, 1] - logits[:, 0] + delta)


if __name__ == "__main__":
    unittest.main()
