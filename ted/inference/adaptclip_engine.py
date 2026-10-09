"""Captured AdaptCLIP image bridge for a hash-verified prepared workspace.

This stage requires the prepared research checkout. Portable bundles and
container/HTTP parity are separate gates; no datasets or fitting are used here.
"""
import importlib.util
import json
from pathlib import Path
import sys
import threading
import time

import torch

from reproduction.checkpoint_download import digest_file
from reproduction.recipe_lookup import execution_recipe
from .adaptclip import CapturedAdaptDensityReadout


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


class CapturedAdaptCLIPEngine:
    def __init__(self, *, export_directory, workspace, device='cpu'):
        started = time.perf_counter()
        exported, workspace = Path(export_directory).resolve(), Path(workspace).resolve()
        manifest_path = exported / 'manifest.json'
        manifest = read(manifest_path)
        if manifest.get('schema_version') != 1 or manifest.get('host') != 'AdaptCLIP':
            raise ValueError('Expected an AdaptCLIP captured-state export')
        source = workspace / 'source'
        files = manifest.get('prepared_source_files', [])
        if not files:
            raise ValueError('Export must bind the prepared evaluator and helper sources')
        for entry in files:
            relative = Path(entry['path'])
            if relative.is_absolute() or '..' in relative.parts or digest_file(source / relative) != entry['sha256']:
                raise ValueError('Prepared AdaptCLIP source differs from captured evaluation')
        for entry in manifest['evidence']:
            if Path(entry['file']).name != entry['file'] or digest_file(exported / entry['file']) != entry['sha256']:
                raise ValueError('Exported AdaptCLIP evaluation evidence changed')
        execution = read(exported / 'execution.json')
        if (execution['status'] != 'matched' or execution.get('returncode') != 0
                or not execution.get('finished') or digest_file(workspace / 'run.json') != execution['plan_sha256']):
            raise ValueError('Prepared workspace does not match a passing terminal execution')
        plan = read(workspace / 'run.json')
        if plan['recipe'] != manifest['recipe'] or execution['recipe'] != manifest['recipe']:
            raise ValueError('Captured AdaptCLIP recipe identity differs')
        summary = read(exported / 'summary.json')['summary']
        catalog = Path(__file__).resolve().parents[2] / 'reproduction'
        recipe = execution_recipe(catalog, manifest['recipe'])
        if recipe['host'] != 'AdaptCLIP':
            raise ValueError('Wrong recipe host')
        dependencies = {entry['sha256'] for entry in manifest['original_model_inputs']}
        binding = next(row for row in read(catalog / 'host-checkpoints.json')['bindings'] if row['recipe'] == recipe['id'])
        checkpoint_hashes = {entry['sha256'] for entry in binding['checkpoint_assets'] if entry['role'] == 'checkpoint_path'}
        checkpoint = Path(summary['checkpoint_path'])
        if len(checkpoint_hashes) != 1 or digest_file(checkpoint) not in checkpoint_hashes or not checkpoint_hashes <= dependencies:
            raise ValueError('AdaptCLIP adapter checkpoint binding differs')
        backbones = read(catalog / 'backbones.json')
        selected = next(row for row in backbones['bindings'] if row['recipe'] == recipe['id'])
        backbone = next(row for row in backbones['artifacts'] if row['sha256'] == selected['sha256'])
        weight = workspace / 'clip-cache' / backbone['filename']
        if digest_file(weight) != backbone['sha256'] or backbone['sha256'] not in dependencies:
            raise ValueError('AdaptCLIP backbone binding differs')
        for name in ['adaptcliplib', 'dataset', 'tools', 'official_parallel_test_adaptclip', 'analyze_adaptclip_vl_score_forms']:
            module = sys.modules.get(name)
            filename = getattr(module, '__file__', None)
            if filename and source not in Path(filename).resolve().parents:
                raise RuntimeError('AdaptCLIP requires an isolated host worker process')
        script = source / 'neurips2026/scripts/official_parallel_test_adaptclip_vlrefine.py'
        spec = importlib.util.spec_from_file_location('ted_captured_adaptclip_host', script)
        self.host = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.host)
        import adaptcliplib.model_load as model_load
        # Preserve the loader's named-model architecture choice. Redirect only
        # its official download to the already hash-verified local object.
        pretrained = summary['pretrained_model']
        if pretrained in model_load._MODELS:
            expected_url = model_load._MODELS[pretrained]
            def pinned_download(url, *args, **kwargs):
                if url != expected_url:
                    raise ValueError('Unexpected AdaptCLIP backbone download request')
                return str(weight)
            model_load._download = pinned_download
            model_load.download_pretrained_from_hf = pinned_download
        elif Path(pretrained).resolve() != weight.resolve():
            raise ValueError('Prepared AdaptCLIP pretrained path differs from verified weight')
        self.device = torch.device(device)
        if self.device.type == 'cuda':
            from reproduction.axis_run import require_live_gpu_allocation
            require_live_gpu_allocation()
        captures = {}
        if manifest['captured_state'] != execution.get('captured_files') and [
            {key: item[key] for key in ['function', 'file', 'sha256']} for item in manifest['captured_state']
        ] != execution.get('captured_files'):
            raise ValueError('AdaptCLIP state differs from terminal capture inventory')
        for entry in manifest['captured_state']:
            if entry['function'] != 'build_vl_rankr_density_calibrator' or entry['object_path'] != 'objects/' + entry['sha256']:
                raise ValueError('Unexpected captured AdaptCLIP state')
            path = exported / entry['object_path']
            if digest_file(path) != entry['sha256']:
                raise ValueError('Captured AdaptCLIP object changed')
            state = torch.load(path, map_location='cpu', weights_only=True)
            observed = {key: value for key, value in state.items() if key not in {'basis', 'fp_coords', 'def_coords'}}
            roles = [role for role in ['vl', 'tl'] if observed == summary['source_' + role + '_calibrator']]
            if len(roles) != 1 or roles[0] in captures:
                raise ValueError('Captured AdaptCLIP branch identity is missing or ambiguous')
            captures[roles[0]] = state
        self.readout = CapturedAdaptDensityReadout(captures, summary, device=device)
        argv = plan['argv']
        reduction = int(argv[argv.index('--vl_reduction') + 1])
        with torch.inference_mode():
            self.model, self.textual, self.visual, self.text, self.dpam_layer = self.host.build_model(
                device=str(self.device), checkpoint_path=str(checkpoint), image_size=summary['image_size'],
                n_ctx=summary['n_ctx'], vl_reduction=reduction, pretrained_model=pretrained,
                checkpoint_load_mode=summary['checkpoint_load_mode'])
        for module in [self.model, self.textual, self.visual]:
            module.requires_grad_(False).eval()
        from tools import get_transform
        self.transform, _ = get_transform(image_size=summary['image_size'])
        self.summary, self.manifest = summary, manifest
        self.artifact_sha256 = digest_file(manifest_path)
        self._lock = threading.Lock()
        self._sync()
        self.model_load_ms = (time.perf_counter() - started) * 1000

    def _sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def info(self):
        return dict(host='AdaptCLIP', recipe=self.manifest['recipe'], backbone=self.manifest['backbone'],
                    input_size=self.summary['image_size'], device=str(self.device),
                    artifact_sha256=self.artifact_sha256, model_load_ms=self.model_load_ms,
                    deployment_status='prepared-workspace research bridge; container parity pending')

    @torch.inference_mode()
    def predict(self, image):
        if image.width * image.height > 25_000_000:
            raise ValueError('Image exceeds pixel limit')
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Inference engine is busy')
        try:
            start = time.perf_counter()
            tensor = self.transform(image.convert('RGB')).unsqueeze(0).to(self.device)
            self._sync()
            preprocessing_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            baseline = self.host.compute_baseline_outputs(self.model, self.textual, self.visual, self.text,
                tensor, self.summary['features_list'], self.dpam_layer, self.summary['image_size'],
                self.summary['sigma'], self.summary['fusion_type'])
            vl = self.host.vl_patch_bank_features(self.visual, baseline['query_patch_feats'])
            tl = self.host.branch_patch_bank_features(self.visual, baseline['query_patch_feats'], 'tl')
            self._sync()
            host_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            host, corrected, host_score, corrected_score = self.readout(vl, tl, baseline['local_vl_map'],
                baseline['local_tl_map'], baseline['global_vl_score'], baseline['global_tl_score'])
            self._sync()
            return dict(host_map=host.cpu().numpy(), cted_map=corrected.cpu().numpy(),
                image_score=float(host_score[0]), cted_image_score=float(corrected_score[0]),
                image_score_policy='official AdaptCLIP global VL/TL and host-map-max average; corrected score uses corrected map max',
                artifact_sha256=self.artifact_sha256,
                timing_ms=dict(preprocessing=preprocessing_ms, host_including_postprocess=host_ms,
                               cted_including_postprocess=(time.perf_counter() - start) * 1000))
        finally:
            self._lock.release()
