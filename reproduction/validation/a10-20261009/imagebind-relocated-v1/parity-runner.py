"""Verify a relocated captured RawImageBind bundle against its original output block."""
import argparse
import ast
import base64
from datetime import datetime, timezone
import json
import io
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
from urllib.parse import urlencode, urlsplit
import urllib.request
from .checkpoint_download import digest_file
from .datasets import load_protocol

def run(workspace, images_root, http_url=None):
    import numpy as np
    import torch
    from PIL import Image
    from ted.inference.imagebind_engine import CapturedRawImageBindEngine
    plan=json.loads((workspace/'run.json').read_text())
    original=Path(plan['cwd']).parent.resolve()
    objects={Path(e['path']).resolve() for e in plan['verified_objects']}
    allowed=None
    if http_url is not None:
        parsed=urlsplit(http_url)
        if (parsed.scheme!='http' or parsed.hostname!='127.0.0.1' or parsed.username or parsed.password
                or parsed.path not in {'','/'} or parsed.query or parsed.fragment):
            raise ValueError('Expected a pinned loopback worker URL')
        allowed=('127.0.0.1',parsed.port or 80)
        http_url=http_url.rstrip('/')
    def guard(event,args):
        if event=='socket.connect' and (allowed is None or args[1]!=allowed):
            raise PermissionError('Network access denied')
        if event=='socket.getaddrinfo' and (allowed is None or (args[0],args[1])!=allowed):
            raise PermissionError('Network resolution denied')
        if event=='open' and isinstance(args[0],(str,bytes)):
            p=Path(args[0].decode() if isinstance(args[0],bytes) else args[0]).resolve()
            if p==original or original in p.parents or p in objects:
                raise PermissionError('Original workspace/model input read denied')
    sys.addaudithook(guard)
    for action in [lambda:open(original/'run.json','rb'),lambda:socket.getaddrinfo('localhost',1)]:
        try:
            action()
        except PermissionError:
            pass
        else:
            raise RuntimeError('Denial self-check failed')
    e=CapturedRawImageBindEngine(export_directory=workspace/'export',workspace=workspace,device='cpu')
    script=workspace/'source/neurips2026/scripts/official_parallel_test_rawimagebind.py'
    tree=ast.parse(script.read_bytes())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    loops=[n for n in ast.walk(main) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple)
           and any(isinstance(t,ast.Name) and t.id=='items' for t in n.target.elts)]
    if len(loops)!=1:
        raise ValueError('Original target inference loop is ambiguous')
    nodes=[]
    recording=False
    for node in loops[0].body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='text_pair':
            recording=True
        if recording:
            if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='append':
                break
            nodes.append(node)
    if not nodes or not any(isinstance(n,ast.Name) and n.id=='calibrated_img_score' for node in nodes for n in ast.walk(node)):
        raise ValueError('Original calibrated output block is incomplete')
    block=compile(ast.Module(body=nodes,type_ignores=[]),str(script),'exec')
    protocol=Path(__file__).resolve().parent/'datasets/btad'
    manifest,metadata=load_protocol(protocol)
    if digest_file(protocol/'manifest.json')!=plan['datasets']['btad']['manifest_sha256']:
        raise ValueError('Canonical target manifest differs from captured execution')
    inputs={(r['role'],r['path']):r for r in manifest['files']}
    rows=[]
    for category,items in metadata['test'].items():
        relative=items[0]['img_path']
        image_path=images_root/relative
        binding=inputs[('images',relative)]
        if digest_file(image_path)!=binding['sha256'] or image_path.stat().st_size!=binding['bytes']:
            raise ValueError('Canonical image bytes changed')
        with Image.open(image_path) as im:
            image=im.convert('RGB')
        if http_url is None:
            actual=e.predict(image,category=category)
        else:
            body=io.BytesIO()
            image.save(body,format='PNG')
            request=urllib.request.Request(http_url+'/predict?'+urlencode({'category':category}),data=body.getvalue(),headers={'Content-Type':'image/png'})
            opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(request,timeout=180) as response:
                actual=json.load(response)
            if actual.get('category')!=category or actual.get('artifact_sha256')!=e.artifact_sha256:
                raise ValueError('HTTP worker changed category or artifact identity')
            with np.load(io.BytesIO(base64.b64decode(actual['maps_npz_base64'])),allow_pickle=False) as maps:
                if set(maps.files)!={'host','tted','cted'}:
                    raise ValueError('HTTP response lost a RawImageBind readout')
                for name in maps.files:
                    actual[name+'_map']=maps[name].copy()
        with torch.inference_mode():
            env=dict(e.host.__dict__,args=SimpleNamespace(**e.summary),backbone=e.backbone,
                image=e.transform(image).unsqueeze(0),cls_name=category,features_list=e.summary['features_list'],
                fp_banks=e.fp,def_banks=e.defect,source_calibrators=e.calibrators,image_size=e.summary['image_size'])
            exec(block,env)
        maps={k:float(np.max(np.abs(np.asarray(actual[k])-np.asarray(env[v])))) for k,v in [('host_map','baseline_map'),('tted_map','parallel_map'),('cted_map','calibrated_map')]}
        scores={k:abs(actual[k]-env[v]) for k,v in [('image_score','baseline_img_score'),('tted_image_score','ours_img_score'),('cted_image_score','calibrated_img_score')]}
        rows.append(dict(category=category,image_sha256=binding['sha256'],map_max_abs_error=maps,score_abs_error=scores))
        print(json.dumps(rows[-1]),flush=True)
    return dict(status='matched' if all(not any(r['map_max_abs_error'].values()) and not any(r['score_abs_error'].values()) for r in rows) else 'mismatch',
        finished=datetime.now(timezone.utc).isoformat(),device='cpu',gpu_used=False,
        recipe=plan['recipe'],artifact_sha256=e.artifact_sha256,
        engine_sha256=digest_file(Path(sys.modules[CapturedRawImageBindEngine.__module__].__file__)),
        original_output_source_sha256=digest_file(script),original_workspace_reads_denied=True,network_denied=http_url is None,
        allowed_http_worker=http_url,
        scope='Three protocol-fixed BTAD images, one per class; captured engine or pinned loopback HTTP worker compared to the original evaluator AST output block on the same CPU node and terminal captured bank/calibrators. No fitting/mining. Original model/workspace reads and unrelated network connections denied.',cases=rows)

def main():
    import torch
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace',type=Path)
    parser.add_argument('images_root',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--http-url')
    args=parser.parse_args()
    torch.set_num_threads(2)
    result=run(args.workspace.resolve(),args.images_root.resolve(),args.http_url)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('TERMINAL',result['status'],flush=True)
    return 0 if result['status']=='matched' else 1

if __name__=='__main__':
    raise SystemExit(main())
