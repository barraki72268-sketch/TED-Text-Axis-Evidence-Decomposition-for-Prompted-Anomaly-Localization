"""Compare portable captured FAPrompt image maps to its original branch output block."""
import argparse
import ast
import base64
from datetime import datetime,timezone
import io
import json
from pathlib import Path
import socket
import sys
from urllib.parse import urlsplit,urlencode
import urllib.request

from .checkpoint_download import digest_file
from .datasets import load_protocol


def run(workspace,images_root,http_url=None):
    import numpy as np
    import torch
    from PIL import Image
    from ted.inference.faprompt_captured_engine import CapturedFAPromptEngine
    exported=workspace/'export';summary=json.loads((exported/'summary.json').read_text());plan=json.loads((workspace/'run.json').read_text())
    allowed=None
    if http_url is not None:
        parsed=urlsplit(http_url)
        if (parsed.scheme!='http' or parsed.hostname!='127.0.0.1' or parsed.username or parsed.password
            or parsed.path not in {'','/'} or parsed.query or parsed.fragment):raise ValueError('Expected pinned loopback worker URL')
        allowed=('127.0.0.1',parsed.port or 80)
        http_url=http_url.rstrip('/')
    original=Path(plan['cwd']).parent.resolve();objects={Path(e['path']).resolve() for e in plan['verified_objects']}
    def guard(event,args):
        if event=='socket.connect' and (allowed is None or args[1]!=allowed):raise PermissionError('Network access denied')
        if event=='socket.getaddrinfo' and (allowed is None or (args[0],args[1])!=allowed):raise PermissionError('Network resolution denied')
        if event=='open' and isinstance(args[0],(str,bytes)):
            p=Path(args[0].decode() if isinstance(args[0],bytes) else args[0]).resolve()
            if p==original or original in p.parents or p in objects:raise PermissionError('Original workspace/model input read denied')
    sys.addaudithook(guard)
    for action in [lambda:open(original/'run.json','rb'),lambda:socket.getaddrinfo('localhost',1)]:
        try:action()
        except PermissionError:pass
        else:raise RuntimeError('Denial self-check failed')
    engine=CapturedFAPromptEngine(workspace=workspace,alpha=summary['alphas'][0],device='cpu')
    script=workspace/'source/neurips2026/scripts/official_parallel_test_faprompt.py';tree=ast.parse(script.read_bytes())
    blocks=[n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=="mode == 'branch_calibrated'"]
    if len(blocks)!=1:raise ValueError('Original output block is ambiguous')
    code=compile(ast.Module(body=blocks[0].body,type_ignores=[]),str(script),'exec')
    protocol=Path(__file__).resolve().parent/'datasets'/summary['target_dataset'];manifest,metadata=load_protocol(protocol)
    if digest_file(protocol/'manifest.json')!=plan['datasets'][summary['target_dataset']]['manifest_sha256']:raise ValueError('Target manifest differs')
    inputs={(e['role'],e['path']):e for e in manifest['files']};rows=[]
    for category,items in metadata['test'].items():
        relative=items[0]['img_path'];p=images_root/relative;binding=inputs[('images',relative)]
        if digest_file(p)!=binding['sha256'] or p.stat().st_size!=binding['bytes']:raise ValueError('Canonical image bytes differ')
        with Image.open(p) as image:im=image.convert('RGB')
        for alpha in summary['alphas']:
            if http_url is None:
                actual=engine.predict_for_alpha(im,alpha)
            else:
                body=io.BytesIO();im.save(body,format='PNG')
                req=urllib.request.Request(http_url+'/predict?'+urlencode({'alpha':alpha}),data=body.getvalue(),headers={'Content-Type':'image/png'})
                opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(req,timeout=120) as response:actual=json.load(response)
                if actual.get('alpha')!=alpha or actual.get('artifact_sha256')!=engine.readout.artifact_sha256:raise ValueError('HTTP worker changed strength or artifact')
                with np.load(io.BytesIO(base64.b64decode(actual['maps_npz_base64'])),allow_pickle=False) as maps:
                    actual['host_map']=maps['host'];actual['cted_map']=maps['cted']
            with torch.inference_mode():
                bank=engine.readout.bank;tensor=engine.transform(im).unsqueeze(0)
                out=engine.host.compute_faprompt_outputs(engine.model,engine.prompts,tensor,summary['features_list'],bank['image_size'],summary['sigma'],summary['dap_token_mode'],bank['dpam_layer'],engine.pair)
                tokens=out['patch_tokens'];axis=bank['axis'];a=engine.readout.artifact;query=tokens @ axis
                qdef=engine.host.logmeanexp_negative_sqdist_1d(query,a['defect_projection'],tau=summary['tau'],chunk=2048)
                qfp=engine.host.logmeanexp_negative_sqdist_1d(query,a['fp_projection'],tau=summary['tau'],chunk=2048)
                ns=dict(engine.host.__dict__,source_branch_calibrators=a['calibrators'],tokens=tokens,q_def=qdef,q_fp=qfp,
                    branch1_token=out['branch1_token_score'],branch2_token=out['branch2_token_score'],baseline_token=out['token_score'],baseline_z=engine.host.spatial_tanh_zscore(out['token_score']),alpha=float(alpha),residual_gate_quantile=summary['residual_gate_quantile'],residual_gate_temp=summary['residual_gate_temp'],image_size=bank['image_size'],sigma=summary['sigma'],out=out)
                exec(code,ns)
            expected_score=float(out['official_image_score'][0])
            row=dict(category=category,alpha=alpha,image_sha256=binding['sha256'],host_max_absolute_error=float(np.abs(actual['host_map']-out['anomaly_map']).max()),cted_max_absolute_error=float(np.abs(actual['cted_map']-ns['maps']).max()),host_image_score_error=abs(actual['image_score']-expected_score),cted_image_score_error=abs(actual['cted_image_score']-expected_score))
            rows.append(row);print(json.dumps(row),flush=True)
    passed=all(r[k]==0 for r in rows for k in ['host_max_absolute_error','cted_max_absolute_error','host_image_score_error','cted_image_score_error'])
    return dict(status='matched' if passed else 'mismatch',rows=rows,engine=engine.info(),denial_self_checks_passed=True,original_input_reads_denied=True,network_denied=http_url is None,allowed_loopback_http=http_url,http_verified=http_url is not None,fitting_performed=False,scope='First canonical test image/class at every recorded alpha. Original baseline functions plus original branch-calibrated AST block; all original workspace/model inputs and network denied. Not a full-dataset CPU metric replay; HTTP verification only if allowed_loopback_http is explicitly set.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('workspace',type=Path);p.add_argument('images_root',type=Path);p.add_argument('--http-url');p.add_argument('--output',type=Path,required=True);a=p.parse_args();report=dict(status='running',started=datetime.now(timezone.utc).isoformat())
    with a.output.open('x') as f:json.dump(report,f)
    try:report.update(run(a.workspace.resolve(),a.images_root.resolve(),a.http_url))
    except Exception as e:report.update(status='failed',error=f'{type(e).__name__}: {e}')
    report['finished']=datetime.now(timezone.utc).isoformat();a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));return 0 if report['status']=='matched' else 1


if __name__=='__main__':raise SystemExit(main())
