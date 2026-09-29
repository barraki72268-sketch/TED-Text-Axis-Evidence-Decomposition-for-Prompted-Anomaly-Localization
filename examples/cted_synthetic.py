"""CPU interface demo with synthetic features, NOT a paper reproduction recipe."""
import argparse
import torch
import torch.nn.functional as F
from ted.cted import aaclip, adaclip, faprompt


def run(host, seed=0, module=None):
    torch.manual_seed(seed)
    module = module or {"aaclip": aaclip, "adaclip": adaclip, "faprompt": faprompt}[host]
    axis = F.normalize(torch.randn(12), dim=0)
    fp = F.normalize(torch.randn(24, 12), dim=-1)
    defect = F.normalize(torch.randn(24, 12) + axis, dim=-1)
    banks = {"fp": fp, "defect": defect,
             "fp_score": fp @ axis, "defect_score": defect @ axis}
    # Native FAPrompt can train distinct branch calibrators. This example
    # exercises one explicit branch, not the complete two-branch fusion.
    banks["fp_branch1_score"] = banks["fp_score"] * 0.8
    banks["defect_branch1_score"] = banks["defect_score"] * 0.8
    kwargs = dict(axis=axis, tau=0.1, max_train_points=32, epochs=3,
                  lr=0.02, rank_margin=0.05, rank_temp=0.05,
                  eta_init=0.05, eta_max=0.2, eta_reg=0.001,
                  fp_weight_init=1.0, learn_fp_weight=True, hardpair_frac=0.75,
                  subspace_rank=4, subspace_basis_control="source",
                  preserve_host_margin_weight=1.0, seed=seed,
                  device=torch.device("cpu"))
    if host == "aaclip":
        cal = module.train_one_layer_subspace_transport_calibrator(
            fp=fp, defect=defect, subspace_readout_mode="host_residual", **kwargs)
    else:
        extra = {"bank_chunk": 8, "host_score_key": "branch1_score"} if host == "faprompt" else {}
        cal = module.train_subspace_host_residual_calibrator(
            banks=banks, pair_label_control="correct", **extra, **kwargs)
    tokens = F.normalize(torch.randn(1, 16, 12), dim=-1)
    query = tokens @ axis
    def support(bank):
        if host == "adaclip":
            return module.logmeanexp_negative_sqdist_1d(query.flatten(), bank @ axis, 0.1).view_as(query)
        if host == "faprompt":
            return module.logmeanexp_negative_sqdist_1d(query, bank @ axis, 0.1, 8)
        return module.logmeanexp_negative_sqdist_1d(query, bank @ axis, 0.1)
    transport = (module.prescore_subspace_transport_seg_tokens if host == "aaclip"
                 else module.prescore_subspace_transport_tokens)
    with torch.no_grad():
        rect = transport(tokens, support(defect), support(fp), cal["basis"],
                         cal["eta"], cal["transport_direction"],
                         cal["transport_b"], cal["fp_weight"])
        if host == "aaclip":
            score = module.subspace_host_residual_map_from_tokens(
                tokens, rect, cal["basis"], cal["subspace_score_w"], axis,
                cal["readout_gamma"], 8, "host_residual")
        else:
            host_score = query * (0.8 if host == "faprompt" else 1.0)
            score = module.subspace_host_residual_token_score(
                host_score, rect, cal["basis"], cal["subspace_score_w"], cal["readout_gamma"])
    assert torch.isfinite(score).all()
    return cal, score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=["all", "aaclip", "adaclip", "faprompt"], default="all")
    args = parser.parse_args()
    for host in ([args.host] if args.host != "all" else ["aaclip", "adaclip", "faprompt"]):
        cal, score = run(host)
        print(f"{host}: finite output {tuple(score.shape)}, rank={cal['subspace_rank']}")
