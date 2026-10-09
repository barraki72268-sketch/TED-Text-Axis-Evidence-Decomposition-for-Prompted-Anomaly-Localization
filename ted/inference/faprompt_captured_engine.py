"""Portable FAPrompt image inference using captured fitted state, without fitting."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import torch

from reproduction.checkpoint_download import digest_file
from reproduction.recipe_lookup import execution_recipe
from .faprompt_captured import CapturedFAPromptReadout


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def argument(script,argv,name):
    if name in argv:
        return argv[argv.index(name)+1]
    nodes=[n for n in ast.walk(ast.parse(script.read_bytes())) if isinstance(n,ast.Call)
           and isinstance(n.func,ast.Attribute) and n.func.attr=='add_argument'
           and any(isinstance(a,ast.Constant) and a.value==name for a in n.args)]
    if len(nodes)!=1:raise ValueError('Original argument default is ambiguous: '+name)
    value=next((k.value for k in nodes[0].keywords if k.arg=='default'),None)
    if not isinstance(value,ast.Constant):raise ValueError('Original argument default is not constant')
    return value.value


class CapturedFAPromptEngine:
    def __init__(self,*,workspace,alpha,device='cpu'):
        started=time.perf_counter();workspace=Path(workspace).resolve();exported=workspace/'export'
        bundle=read(workspace/'serving-bundle.json');manifest=read(exported/'manifest.json')
        execution=read(exported/'execution.json');summary=read(exported/'summary.json');plan=read(workspace/'run.json')
        if (bundle.get('host')!='FAPrompt' or bundle.get('schema_version')!=1
            or bundle['recipe']!=manifest['recipe'] or bundle['export_sha256']!=digest_file(exported/'manifest.json')
            or digest_file(workspace/'run.json')!=execution['plan_sha256']
            or bundle['original_plan_sha256']!=execution['plan_sha256'] or plan['recipe']!=manifest['recipe']):
            raise ValueError('Bundle does not bind captured FAPrompt execution')
        source=workspace/'source'
        for entry in bundle['files']:
            rel=Path(entry['path']);p=workspace/rel
            if rel.is_absolute() or '..' in rel.parts or p.is_symlink() or digest_file(p)!=entry['sha256'] or p.stat().st_size!=entry['bytes']:
                raise ValueError('Portable FAPrompt input changed')
        for entry in manifest['prepared_source_files']:
            rel=Path(entry['path'])
            if rel.is_absolute() or '..' in rel.parts or digest_file(source/rel)!=entry['sha256']:
                raise ValueError('Prepared research source changed')
        catalog=Path(__file__).resolve().parents[2]/'reproduction';recipe=execution_recipe(catalog,manifest['recipe'])
        if recipe['host']!='FAPrompt':raise ValueError('Wrong recipe host')
        dependencies={e['sha256'] for e in manifest['original_model_inputs']}
        binding=next(b for b in read(catalog/'host-checkpoints.json')['bindings'] if b['recipe']==recipe['id'])
        checkpoint=workspace/'checkpoints/epoch_15.pth';hashes={b['sha256'] for b in binding['checkpoint_assets'] if b['role']=='checkpoint_path'}
        if len(hashes)!=1 or digest_file(checkpoint) not in hashes or not hashes<=dependencies:
            raise ValueError('Checkpoint binding differs')
        backbones=read(catalog/'backbones.json');selected=next(b for b in backbones['bindings'] if b['recipe']==recipe['id'])
        backbone=next(b for b in backbones['artifacts'] if b['sha256']==selected['sha256']);weight=workspace/'clip-cache'/backbone['filename']
        if digest_file(weight)!=backbone['sha256'] or backbone['sha256'] not in dependencies:
            raise ValueError('Backbone binding differs')
        self.device=torch.device(device)
        self.readout=CapturedFAPromptReadout(exported,alpha=alpha,device=device)
        for name in ['AnomalyCLIP_lib','FAPrompt','utils','common_failure_metrics']:
            old=sys.modules.get(name);filename=getattr(old,'__file__',None)
            if filename and source not in Path(filename).resolve().parents:
                raise RuntimeError('FAPrompt requires an isolated worker process')
        os.environ['FAPROMPT_CACHE_DIR']=str(workspace/'clip-cache')
        script=source/'neurips2026/scripts/official_parallel_test_faprompt.py'
        spec=importlib.util.spec_from_file_location('ted_captured_faprompt_host',script)
        self.host=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.host)
        import AnomalyCLIP_lib.model_load as model_load
        if summary['model_name'] not in model_load._MODELS:raise ValueError('Expected recorded named-model loader')
        model_load._MODELS[summary['model_name']]=str(weight)
        def no_download(*args,**kwargs):raise RuntimeError('Network download prohibited in captured inference')
        model_load._download=no_download
        # Match main()'s initialization order, including projected prompt tails.
        self.host.setup_seed(int(argument(script,plan['argv'],'--seed')))
        with torch.inference_mode():
            self.model,self.prompts,self.prompt_info=self.host.build_model(self.device,str(checkpoint),
                summary['model_name'],summary['prompt_load_mode'],self.readout.bank['dpam_layer'],
                int(argument(script,plan['argv'],'--depth')),int(argument(script,plan['argv'],'--n_ctx')),int(argument(script,plan['argv'],'--t_n_ctx')))
            self.pair=self.host.learned_text_pair(self.model,self.prompts)
            axis=torch.nn.functional.normalize(self.pair[1]-self.pair[0],dim=0).cpu()
            torch.testing.assert_close(axis,self.readout.bank['axis'],atol=1e-5,rtol=1e-5)
        self.transform,_=self.host.get_transform(SimpleNamespace(image_size=self.readout.bank['image_size']))
        if summary['ours_image_score_mode']!='official':raise ValueError('Non-official image-score policy needs separate implementation')
        self.summary=summary;self.manifest=manifest;self.alpha=float(alpha);self._lock=threading.Lock()
        self.model_load_ms=(time.perf_counter()-started)*1000

    def info(self):
        return dict(host='FAPrompt',recipe=self.manifest['recipe'],backbone=self.manifest['backbone'],alpha=self.alpha,
            artifact_sha256=self.readout.artifact_sha256,device=str(self.device),portable_bundle=True,
            deployment_status='captured-state image bridge; image/HTTP verification separate',model_load_ms=self.model_load_ms)

    @torch.inference_mode()
    def predict(self,image):
        if image.width*image.height>25_000_000:raise ValueError('Image exceeds pixel limit')
        if not self._lock.acquire(blocking=False):raise RuntimeError('Inference engine is busy')
        try:
            start=time.perf_counter();bank=self.readout.bank;s=self.summary
            tensor=self.transform(image.convert('RGB')).unsqueeze(0).to(self.device)
            out=self.host.compute_faprompt_outputs(self.model,self.prompts,tensor,s['features_list'],bank['image_size'],s['sigma'],s['dap_token_mode'],bank['dpam_layer'],self.pair)
            corrected=self.readout(out['patch_tokens'],out['token_score'],out['branch1_token_score'],out['branch2_token_score'])
            maps=self.host.upsample_tokens(corrected,bank['image_size'],s['sigma'])
            score=float(out['official_image_score'][0])
            return dict(host_map=out['anomaly_map'],cted_map=maps,image_score=score,cted_image_score=score,
                image_score_policy='recorded official FAPrompt image score for both host and C-TED; corrected map uses explicit recorded alpha',
                artifact_sha256=self.readout.artifact_sha256,timing_ms=dict(total=(time.perf_counter()-start)*1000))
        finally:self._lock.release()
