"""Read captured FAPrompt branch state without mining banks or fitting calibrators."""
import json
from pathlib import Path
import torch

from reproduction.checkpoint_download import digest_file
from .faprompt import BranchCalibratedReadout


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


class CapturedFAPromptReadout:
    def __init__(self, export_directory, *, alpha, device='cpu', bank_chunk=2048):
        exported = Path(export_directory).resolve()
        manifest = read(exported / 'manifest.json')
        if manifest.get('host') != 'FAPrompt' or manifest.get('schema_version') != 1:
            raise ValueError('Expected a FAPrompt captured-state export')
        for entry in manifest['evidence']:
            if Path(entry['file']).name != entry['file'] or digest_file(exported / entry['file']) != entry['sha256']:
                raise ValueError('Captured evaluation evidence changed')
        execution = read(exported / 'execution.json')
        summary = read(exported / 'summary.json')
        if (execution.get('status') != 'matched' or execution.get('returncode') != 0
                or not execution.get('finished') or execution['recipe'] != manifest['recipe']
                or not execution['comparison']['all_match_2dp']):
            raise ValueError('Expected a matching terminal evaluation')
        if summary['insert_modes'] != ['branch_calibrated'] or float(alpha) not in summary['alphas']:
            raise ValueError('Strength/mode must be explicitly recorded in this execution')
        bank, branches = None, {}
        inventory = read(exported / 'capture-index.json')
        if digest_file(exported / 'capture-index.json') != execution['capture_index_sha256']:
            raise ValueError('Terminal capture index binding changed')
        for entry in manifest['captured_state']:
            bound = {k: entry[k] for k in ['function', 'file', 'sha256']}
            if bound not in inventory or bound not in execution['captured_files']:
                raise ValueError('Capture differs from terminal inventory')
            if entry['object_path'] != 'objects/' + entry['sha256']:
                raise ValueError('Unexpected captured object path')
            path = exported / entry['object_path']
            if digest_file(path) != entry['sha256']:
                raise ValueError('Captured state bytes changed')
            state = torch.load(path, map_location='cpu', weights_only=True)
            if entry['function'] == 'get_or_collect_source_banks':
                if bank is not None:
                    raise ValueError('Duplicate source bank')
                bank = state
            elif entry['function'] == 'train_subspace_host_residual_calibrator':
                role = {'faprompt_branch1_score': 'branch1', 'faprompt_branch2_score': 'branch2'}.get(state['host_score_source'])
                if role is None or role in branches or any(state[k] != v for k,v in summary['source_branch_calibrators'][role].items()):
                    raise ValueError('Captured branch identity differs from recorded metadata')
                branches[role] = state
            else:
                raise ValueError('Unexpected fitted state')
        if bank is None or set(branches) != {'branch1','branch2'}:
            raise ValueError('Incomplete bank or branch state')
        self.artifact = dict(schema_version=1,host='faprompt',axis=bank['axis'],
            fp_projection=bank['fp'].float() @ bank['axis'],
            defect_projection=bank['defect'].float() @ bank['axis'],calibrators=branches,
            recipe=dict(tau=summary['tau']),inference=dict(alpha=float(alpha),
                gate_quantile=summary['residual_gate_quantile'],gate_temp=summary['residual_gate_temp']))
        self.readout = BranchCalibratedReadout(self.artifact,device=device,bank_chunk=bank_chunk)
        self.manifest,self.summary,self.bank = manifest,summary,bank
        self.artifact_sha256 = digest_file(exported / 'manifest.json')
        self.alpha = float(alpha)

    def __call__(self,tokens,baseline,branch1,branch2):
        return self.readout(tokens,baseline,branch1,branch2)
