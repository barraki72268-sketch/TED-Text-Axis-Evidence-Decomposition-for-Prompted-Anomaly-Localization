"""Check real HTTP maps/scores against the original AdaptCLIP CPU output block."""
import argparse
import base64
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit
import urllib.request

from .adaptclip_image_parity import original_output_block
from .checkpoint_download import digest_file
from .datasets import load_protocol


def run(exported, workspace, images_root, url):
    import numpy as np
    from PIL import Image
    import torch
    from ted.inference.adaptclip_engine import CapturedAdaptCLIPEngine
    parsed=urlsplit(url)
    if parsed.scheme!='http' or parsed.hostname not in {'127.0.0.1','localhost'} or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('Use an explicit local HTTP worker endpoint')
    engine=CapturedAdaptCLIPEngine(export_directory=exported,workspace=workspace,device='cpu')
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url+'/model-info',timeout=10) as response:
        worker=json.load(response)
    if worker['artifact_sha256']!=engine.artifact_sha256 or worker['device']!='cpu':
        raise ValueError('Worker artifact or device differs')
    root=Path(__file__).resolve().parent
    protocol=root/'datasets'/engine.summary['target_dataset']
    manifest,metadata=load_protocol(protocol)
    plan=json.loads((workspace/'run.json').read_text())
    if digest_file(protocol/'manifest.json')!=plan['datasets'][engine.summary['target_dataset']]['manifest_sha256']:
        raise ValueError('Target input protocol changed')
    inputs={(r['role'],r['path']):r for r in manifest['files']}
    captures={}
    for entry in engine.manifest['captured_state']:
        value=torch.load(exported/entry['object_path'],map_location='cpu',weights_only=True)
        observed={k:v for k,v in value.items() if k not in {'basis','fp_coords','def_coords'}}
        roles=[r for r in ['vl','tl'] if observed==engine.summary['source_'+r+'_calibrator']]
        if len(roles)!=1 or roles[0] in captures:
            raise ValueError('Reference calibrator branch differs')
        captures[roles[0]]=value
    code=original_output_block(workspace/'source/neurips2026/scripts/official_parallel_test_adaptclip_vlrefine.py')
    rows=[]
    for category,items in metadata['test'].items():
        relative=items[0]['img_path']; image_path=images_root/relative
        binding=inputs[('images',relative)]
        if digest_file(image_path)!=binding['sha256'] or image_path.stat().st_size!=binding['bytes']:
            raise ValueError('Canonical fixture bytes changed')
        with Image.open(image_path) as opened:
            image=opened.convert('RGB')
        body=io.BytesIO();image.save(body,format='PNG')
        request=urllib.request.Request(url+'/predict',data=body.getvalue(),headers={'Content-Type':'image/png'})
        with opener.open(request,timeout=120) as response:
            actual=json.load(response)
        if actual['artifact_sha256']!=engine.artifact_sha256:
            raise ValueError('HTTP response artifact differs')
        with np.load(io.BytesIO(base64.b64decode(actual['maps_npz_base64'])),allow_pickle=False) as maps:
            host_actual,cted_actual=maps['host'].copy(),maps['cted'].copy()
        with torch.inference_mode():
            tensor=engine.transform(image).unsqueeze(0)
            cfg,host=engine.summary,engine.host
            baseline=host.compute_baseline_outputs(engine.model,engine.textual,engine.visual,engine.text,
                tensor,cfg['features_list'],engine.dpam_layer,cfg['image_size'],cfg['sigma'],cfg['fusion_type'])
            vl=host.vl_patch_bank_features(engine.visual,baseline['query_patch_feats'])
            tl=host.branch_patch_bank_features(engine.visual,baseline['query_patch_feats'],'tl')
            namespace=dict(vars(host),args=SimpleNamespace(**cfg,device='cpu'),image=tensor,
                baseline=baseline,vl_patch=vl,tl_patch=tl,h=math.isqrt(vl.shape[0]),
                vl_axis=None,tl_axis=None,fp_coeff=None,def_coeff=None,tl_fp_coeff=None,tl_def_coeff=None,
                source_vl_calibrator=captures['vl'],source_tl_calibrator=captures['tl'],
                vl_map=host.resize_map(host.smooth_map(baseline['local_vl_map'],cfg['sigma']),cfg['image_size']),
                tl_map=host.resize_map(host.smooth_map(baseline['local_tl_map'],cfg['sigma']),cfg['image_size']),
                raw_tl_map=baseline['local_tl_map'],outputs_to_append=[])
            exec(code,namespace)
            _,expected_map,expected_score=namespace['outputs_to_append'][0]
        expected_host=baseline['baseline_map'].numpy();expected_cted=expected_map.numpy()
        if host_actual.shape!=expected_host.shape or cted_actual.shape!=expected_cted.shape:
            raise ValueError('HTTP map shape differs')
        rows.append(dict(category=category,image=relative,image_sha256=binding['sha256'],
            artifact_sha256=engine.artifact_sha256,map_shape=list(host_actual.shape),
            host_max_absolute_error=float(np.max(np.abs(host_actual-expected_host))),
            cted_max_absolute_error=float(np.max(np.abs(cted_actual-expected_cted))),
            host_image_score_error=abs(actual['image_score']-float(baseline['baseline_img_score'][0])),
            cted_image_score_error=abs(actual['cted_image_score']-float(expected_score[0]))))
    passed=all(r[k]==0 for r in rows for k in ['host_max_absolute_error','cted_max_absolute_error','host_image_score_error','cted_image_score_error'])
    return dict(status='matched' if passed else 'mismatch',rows=rows,worker=worker,device='cpu',
        model_fitting=False,reference='Original baseline functions and calibrated-output AST block on same-host CPU; independent from HTTP serialization and adapter readout.',
        scope='Three preselected canonical BTAD images; not full-dataset CPU metric reproduction or standalone-image verification.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('export_directory',type=Path)
    parser.add_argument('workspace',type=Path)
    parser.add_argument('images_root',type=Path)
    parser.add_argument('url')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=dict(status='running',started=datetime.now(timezone.utc).isoformat())
    with args.output.open('x',encoding='utf-8') as output:
        json.dump(report,output,indent=2)
    try:
        report.update(run(args.export_directory.resolve(),args.workspace.resolve(),args.images_root.resolve(),args.url.rstrip('/')))
    except Exception as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}')
    report['finished']=datetime.now(timezone.utc).isoformat()
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    return 0 if report['status']=='matched' else 1


if __name__=='__main__':
    raise SystemExit(main())
