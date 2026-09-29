"""Run from the repository root: python -m examples.tted_synthetic."""
import torch
from ted import text_axis, tted_score

torch.manual_seed(0)
patches = torch.randn(64, 32)
defect_bank = torch.randn(128, 32)
hard_fp_bank = torch.randn(128, 32)
axis = text_axis(torch.randn(32), torch.randn(32))
with torch.no_grad():
    margin = tted_score(patches, defect_bank, hard_fp_bank, axis, tau=0.05)
assert margin.shape == (64,) and torch.isfinite(margin).all()
print("Synthetic T-TED smoke test passed: 64 finite patch scores.")
print("Random features only; this does not reproduce a paper result.")
