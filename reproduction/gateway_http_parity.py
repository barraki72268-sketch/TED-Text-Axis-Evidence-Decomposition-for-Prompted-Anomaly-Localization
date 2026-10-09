"""Compare each registered local worker with the model gateway on canonical images."""
import argparse
import base64
from datetime import datetime,timezone
import io
import json
from pathlib import Path
from urllib.parse import urlencode,urlsplit
import urllib.request
import urllib.error

from .checkpoint_download import digest_file
from .datasets import load_protocol


def run(registry_path,images_root,gateway):
    import numpy as np
    from PIL import Image
    registry=json.loads(registry_path.read_text())['models']
    manifest,metadata=load_protocol(Path(__file__).resolve().parent/'datasets/btad')
    inputs={(r['role'],r['path']):r for r in manifest['files']}
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for url in [gateway]+[r['url'] for r in registry]:
        p=urlsplit(url)
        if p.scheme!='http' or p.hostname not in {'127.0.0.1','localhost'} or p.username or p.password or p.query or p.fragment:
            raise ValueError('Expected local worker/gateway URL')

    def request(url,body=None):
        req=urllib.request.Request(url,data=body,headers={'Content-Type':'image/png'} if body is not None else {})
        with opener.open(req,timeout=120) as response:
            return json.load(response),dict(response.headers)

    rows=[]
    for model in registry:
        info,_=request(gateway+'/model-info?'+urlencode({'model':model['id']}))
        if info['artifact_sha256']!=model['artifact_sha256']:
            raise ValueError('Gateway worker identity differs')
        for category,items in metadata['test'].items():
            relative=items[0]['img_path'];path=images_root/relative;binding=inputs[('images',relative)]
            if digest_file(path)!=binding['sha256'] or path.stat().st_size!=binding['bytes']:
                raise ValueError('Canonical fixture differs')
            with Image.open(path) as image:
                buffer=io.BytesIO();image.convert('RGB').save(buffer,format='PNG')
            body=buffer.getvalue();params={'category':category} if info.get('categories') else {}
            expected,_=request(model['url']+'/predict'+('?' + urlencode(params) if params else ''),body)
            actual,headers=request(gateway+'/predict?'+urlencode(dict(params,model=model['id'])),body)
            if (actual['artifact_sha256']!=expected['artifact_sha256'] or actual['artifact_sha256']!=model['artifact_sha256']
                    or headers.get('X-TED-Model',headers.get('x-ted-model'))!=model['id']):
                raise ValueError('Routed prediction identity differs')
            with np.load(io.BytesIO(base64.b64decode(actual['maps_npz_base64'])),allow_pickle=False) as a, np.load(io.BytesIO(base64.b64decode(expected['maps_npz_base64'])),allow_pickle=False) as e:
                if a['host'].shape!=e['host'].shape or a['cted'].shape!=e['cted'].shape:
                    raise ValueError('Gateway changed map shape')
                row=dict(model=model['id'],fixture_category=category,image_sha256=binding['sha256'],
                    artifact_sha256=actual['artifact_sha256'],host_max_abs_error=float(np.max(np.abs(a['host']-e['host']))),
                    cted_max_abs_error=float(np.max(np.abs(a['cted']-e['cted']))),
                    raw_image_score_abs_error=abs(actual['image_score']-expected['image_score']))
            if ('cted_image_score' in expected)!=('cted_image_score' in actual):
                raise ValueError('Gateway changed corrected-score availability')
            if 'cted_image_score' in expected:
                row['cted_image_score_abs_error']=abs(actual['cted_image_score']-expected['cted_image_score'])
            rows.append(row)
            print(json.dumps(row),flush=True)
    def rejected(url):
        try:
            request(url,body)
        except urllib.error.HTTPError as error:
            return error.code
        return 200
    unknown=rejected(gateway+'/predict?model=not-registered')
    missing=rejected(gateway+'/predict?model=aaclip-l336-main')
    passed=all(r[k]==0 for r in rows for k in ['host_max_abs_error','cted_max_abs_error','raw_image_score_abs_error']) and all(r.get('cted_image_score_abs_error',0)==0 for r in rows) and unknown==404 and missing==422
    return dict(status='matched' if passed else 'mismatch',models=len(registry),cases=len(rows),rows=rows,
        unknown_model_status=unknown,missing_aa_category_status=missing,registry_sha256=digest_file(registry_path),
        scope='Every registered worker on three fixed canonical BTAD images; direct-versus-gateway raw maps and scores. Not full-dataset metric reproduction or standalone-image verification.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('registry',type=Path)
    parser.add_argument('images_root',type=Path)
    parser.add_argument('gateway')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=dict(status='running',started=datetime.now(timezone.utc).isoformat())
    with args.output.open('x',encoding='utf-8') as out:
        json.dump(report,out,indent=2)
    try:
        report.update(run(args.registry.resolve(),args.images_root.resolve(),args.gateway.rstrip('/')))
    except Exception as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}')
    report['finished']=datetime.now(timezone.utc).isoformat()
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    return 0 if report['status']=='matched' else 1


if __name__=='__main__':
    raise SystemExit(main())
