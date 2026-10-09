"""Compare captured FAPrompt token readout to unchanged original functions on fixed tensors.

This validates readout equations with real captured banks; it is not image/model parity.
"""
import argparse
import ast
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import torch
import torch.nn.functional as F

from .checkpoint_download import digest_file
from ted.inference.faprompt_captured import CapturedFAPromptReadout


def run(exported,script):
    manifest=json.loads((exported/'manifest.json').read_text())
    entry=next(e for e in manifest['prepared_source_files'] if e['path']=='neurips2026/scripts/official_parallel_test_faprompt.py')
    if digest_file(script)!=entry['sha256']:
        raise ValueError('Reference evaluator bytes differ from captured execution')
    functions=['spatial_tanh_zscore','logmeanexp_negative_sqdist_1d','prescore_subspace_transport_tokens','subspace_host_residual_token_score','high_score_gate_tokens']
    tree=ast.parse(script.read_bytes());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in functions]
    if {n.name for n in nodes}!=set(functions):raise ValueError('Original function definitions missing')
    ns=dict(torch=torch,F=F,math=math)
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(script),'exec'),ns)
    summary=json.loads((exported/'summary.json').read_text());rows=[]
    for alpha in summary['alphas']:
        readout=CapturedFAPromptReadout(exported,alpha=alpha,device='cpu')
        gen=torch.Generator(device='cpu').manual_seed(1729)
        tokens=F.normalize(torch.randn(2,35,len(readout.bank['axis']),generator=gen),dim=-1)
        scores=[torch.rand(2,35,generator=gen) for _ in range(3)]
        artifact=readout.artifact;query=tokens @ artifact['axis']
        qdef=ns['logmeanexp_negative_sqdist_1d'](query,artifact['defect_projection'],tau=summary['tau'],chunk=2048)
        qfp=ns['logmeanexp_negative_sqdist_1d'](query,artifact['fp_projection'],tau=summary['tau'],chunk=2048)
        corrected=[]
        for branch,anchor in zip(['branch1','branch2'],scores[1:]):
            c=artifact['calibrators'][branch]
            rect=ns['prescore_subspace_transport_tokens'](tokens=tokens,q_def=qdef,q_fp=qfp,basis=c['basis'],eta=float(c['eta']),direction=c['transport_direction'],transport_b=float(c['transport_b']),fp_weight=float(c['fp_weight']))
            corrected.append(ns['subspace_host_residual_token_score'](host_score=anchor,rect_tokens=rect,basis=c['basis'],score_w=c['subspace_score_w'],readout_gamma=float(c.get('readout_gamma',1.0))))
        gate=ns['high_score_gate_tokens'](ns['spatial_tanh_zscore'](scores[0]),topk_frac=max(1e-6,1-float(summary['residual_gate_quantile'])),sharpness=max(1e-6,1/max(float(summary['residual_gate_temp']),1e-6)))
        expected=scores[0]+float(alpha)*gate*.5*((corrected[0]-scores[1])+(corrected[1]-scores[2]))
        actual=readout(tokens,*scores)
        error=float((actual-expected).abs().max())
        torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-7)
        rows.append(dict(alpha=alpha,max_absolute_error=error,shape=list(actual.shape)))
    return dict(status='matched_within_declared_tolerance',rtol=1e-6,atol=1e-7,recipe=manifest['recipe'],export_sha256=digest_file(exported/'manifest.json'),original_evaluator_sha256=entry['sha256'],rows=rows,fitting_performed=False,image_inference_verified=False,scope='Fixed synthetic normalized tokens/scores with actual captured source bank and branch calibrators; original function AST evaluated independently. Does not verify model preprocessing or image maps.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('exported',type=Path);p.add_argument('script',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    report=dict(status='running',started=datetime.now(timezone.utc).isoformat())
    with a.output.open('x') as f:json.dump(report,f)
    try:report.update(run(a.exported,a.script))
    except Exception as e:report.update(status='failed',error=f'{type(e).__name__}: {e}')
    report['finished']=datetime.now(timezone.utc).isoformat();a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));return 0 if report['status']=='matched_within_declared_tolerance' else 1


if __name__=='__main__':raise SystemExit(main())
