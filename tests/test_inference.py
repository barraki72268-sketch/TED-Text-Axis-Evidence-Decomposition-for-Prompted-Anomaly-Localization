import io
import math
import unittest

import torch
import torch.nn.functional as F

from ted.cted import faprompt as core
from ted.inference.faprompt import BranchCalibratedReadout, export_artifact


def fixture():
    torch.manual_seed(8)
    axis = F.normalize(torch.randn(12), dim=0)
    banks = dict(axis=axis, model_name="synthetic", prompt_load_mode="strict",
                 features_list=[1], image_size=16, dap_token_mode="official_firstk",
                 dpam_layer=1, protocol_version="synthetic")
    for role in ("fp", "defect"):
        banks[role] = F.normalize(torch.randn(16, 12), dim=-1)
        for branch in (1, 2):
            banks[f"{role}_branch{branch}_score"] = banks[role] @ axis * branch
    artifact = export_artifact(banks, checkpoint_sha256="synthetic",
                               research_sha256="synthetic", alpha=1., epochs=2,
                               max_train_points=16)
    tokens = F.normalize(torch.randn(2, 16, 12), dim=-1)
    a, b = torch.rand(2, 16), torch.rand(2, 16)
    return banks, artifact, (tokens, 0.5 * (a + b), a, b)


def reference(banks, artifact, inputs):
    """Uncached composition of the released, source-parity-tested primitives."""
    tokens, baseline, a, b = inputs
    axis = artifact["axis"]
    qd = core.logmeanexp_negative_sqdist_1d(tokens @ axis, banks["defect"] @ axis, 0.1, 2048)
    qf = core.logmeanexp_negative_sqdist_1d(tokens @ axis, banks["fp"] @ axis, 0.1, 2048)
    residuals = []
    for name, anchor in (("branch1", a), ("branch2", b)):
        cal = artifact["calibrators"][name]
        rect = core.prescore_subspace_transport_tokens(tokens, qd, qf, cal["basis"],
            cal["eta"], cal["transport_direction"], cal["transport_b"], cal["fp_weight"])
        corrected = core.subspace_host_residual_token_score(anchor, rect, cal["basis"],
            cal["subspace_score_w"], cal["readout_gamma"])
        residuals.append(corrected - anchor)
    z = core.spatial_tanh_zscore(baseline)
    k = max(1, math.ceil(z.shape[1] * (1. - artifact["inference"]["gate_quantile"])))
    threshold = torch.topk(z, k=k, dim=1, largest=True).values[:, -1:]
    gate = torch.sigmoid((1. / artifact["inference"]["gate_temp"]) * (z - threshold)
                         / z.std(dim=1, keepdim=True).clamp_min(1e-6))
    return baseline + artifact["inference"]["alpha"] * gate * 0.5 * sum(residuals)


class InferenceTests(unittest.TestCase):
    def test_cached_readout_parity_and_chunking(self):
        banks, artifact, inputs = fixture()
        expected = reference(banks, artifact, inputs)
        for chunk in (1, 7, 2048):
            output = BranchCalibratedReadout(artifact, bank_chunk=chunk)(*inputs)
            torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-6)
            self.assertFalse(output.requires_grad)

    def test_serialization_roundtrip(self):
        _, artifact, inputs = fixture()
        stream = io.BytesIO()
        torch.save(artifact, stream)
        stream.seek(0)
        loaded = torch.load(stream, weights_only=True)
        torch.testing.assert_close(BranchCalibratedReadout(artifact)(*inputs),
                                   BranchCalibratedReadout(loaded)(*inputs))

    def test_zero_strength_preserves_host(self):
        _, artifact, inputs = fixture()
        artifact["inference"]["alpha"] = 0.
        torch.testing.assert_close(BranchCalibratedReadout(artifact)(*inputs), inputs[1])

    def test_invalid_inputs_fail(self):
        _, artifact, inputs = fixture()
        runtime = BranchCalibratedReadout(artifact)
        with self.assertRaises(ValueError):
            runtime(inputs[0][:, :1], *(x[:, :1] for x in inputs[1:]))
        with self.assertRaises(ValueError):
            runtime(inputs[0] * float("nan"), *inputs[1:])
        with self.assertRaises(ValueError):
            runtime(inputs[0], inputs[1][:, :3], *inputs[2:])
        with self.assertRaises(ValueError):
            BranchCalibratedReadout(artifact, bank_chunk=0)

    def test_no_host_score_fallback_on_export(self):
        banks, _, _ = fixture()
        del banks["fp_branch2_score"]
        with self.assertRaises(KeyError):
            export_artifact(banks, checkpoint_sha256="test", research_sha256="test", alpha=1.)


if __name__ == "__main__":
    unittest.main()
