import ast,hashlib,importlib.util,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
base=Path('/mnt/data/pilab-kingjinyoung/ted-reproduce-20261008')
sys.path.insert(0,str(base/'github-bayes-rtx-20261010'))
adapter=base/'reports/bayespfl_captured-20261010-v1.py'
spec=importlib.util.spec_from_file_location('bayes_readout',adapter);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
export=base/'exports/bayespfl-bplus-btad-20261010-v1'
readout=module.CapturedBayesPFLReadout(export)
workspace=base/'rtx-bayes-runs-20261010/bayespfl-vitb_plus-mvtec2btad-seed0'
relative='neurips2026/scripts/probe_bayespfl_ted_smoke_20260501.py'
source=workspace/'source'/relative
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(source)==next(x['sha256'] for x in readout.manifest['prepared_source_files'] if x['path']==relative)
names={'logmeanexp_negative_sqdist_1d','compute_ted_layer_scores','resize_patch_score','combine_calibrated_ted_with_host'}
nodes=[x for x in ast.parse(source.read_text()).body if isinstance(x,ast.FunctionDef) and x.name in names]
assert len(nodes)==4
namespace={'torch':torch,'np':np,'F':F};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
torch.manual_seed(20261010)
outputs=[]
for layer in sorted(readout.fp):
 dim=readout.fp[layer].shape[1]
 outputs.append({'dense_feature':F.normalize(torch.randn(49,dim),dim=-1),'axis':F.normalize(torch.randn(dim),dim=0)})
baseline=np.random.default_rng(20261010).random((readout.args.image_size,readout.args.image_size),dtype=np.float32)
expected=namespace['combine_calibrated_ted_with_host'](readout.args,baseline,outputs,readout.fp,readout.defect,readout.calibrators,torch.device('cpu'))
actual=readout(baseline,outputs)
error=float(np.max(np.abs(expected-actual)))
record={'status':'matched' if error==0 else 'mismatch','scope':'Captured real bank/calibrators and fixed synthetic layer features; independent original AST equations, no fitting; not image inference or deployment','max_absolute_map_error':error,'layers':len(outputs),'image_size':readout.args.image_size,'export_manifest_sha256':sha(export/'manifest.json'),'adapter_sha256':sha(adapter),'original_script_sha256':sha(source),'image_inference_verified':False,'deployment_ready':False}
(base/'reports/bayes-readout-parity-20261010-v1.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record));assert error==0
