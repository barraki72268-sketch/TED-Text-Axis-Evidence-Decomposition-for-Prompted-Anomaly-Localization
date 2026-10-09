"""Experimental captured AA host bridge. Run in an isolated host worker.

The verified prepared research workspace is still needed. The engine only
encodes prompts/images and reads captured calibrators; it never loads datasets
or fits state. This is not yet the portable all-model Docker release.
"""
import importlib.util
import json
from pathlib import Path
import sys
import threading
import time

import torch
import torch.nn.functional as F

from reproduction.checkpoint_download import digest_file
from reproduction.recipe_lookup import execution_recipe
from .aaclip import CapturedAAReadout


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


class CapturedAAEngine:
    def __init__(self, *, export_directory, workspace, device='cpu', category=None):
        started = time.perf_counter()
        exported, workspace = Path(export_directory).resolve(), Path(workspace).resolve()
        manifest_path = exported / 'manifest.json'
        manifest = read(manifest_path)
        if manifest.get('schema_version') != 1 or manifest.get('host') != 'AA-CLIP':
            raise ValueError('Expected an AA-CLIP captured-state export')
        source = workspace / 'source'
        files = manifest.get('prepared_source_files', [])
        if not files:
            raise ValueError('Export must bind the prepared evaluator and helper source files')
        for entry in files:
            relative = Path(entry['path'])
            if relative.is_absolute() or '..' in relative.parts or digest_file(source / relative) != entry['sha256']:
                raise ValueError('Prepared source differs from captured evaluation')
        for entry in manifest['evidence']:
            if Path(entry['file']).name != entry['file'] or digest_file(exported / entry['file']) != entry['sha256']:
                raise ValueError('Exported evaluation evidence changed')
        execution = read(exported / 'execution.json')
        if execution['status'] != 'matched' or digest_file(workspace / 'run.json') != execution['plan_sha256']:
            raise ValueError('Workspace does not match the passing captured execution')
        summary = read(exported / 'summary.json')
        if summary['blend_modes'].count('prescore_calibrated') != 1 or 1.0 not in summary['alphas']:
            raise ValueError('Expected the fixed prescore_calibrated_alpha_1 comparison recipe')
        catalog = Path(__file__).resolve().parents[2] / 'reproduction'
        recipe = execution_recipe(catalog, manifest['recipe'])
        binding_id = recipe.get('dependency_recipe', recipe['id'])
        checkpoints = read(catalog / 'host-checkpoints.json')
        binding = next(b for b in checkpoints['bindings'] if b['recipe'] == binding_id)
        ckpt_dir = Path(summary['ckpt_dir'])
        for entry in binding['checkpoint_assets']:
            if digest_file(ckpt_dir / entry['checkpoint_filename']) != entry['sha256']:
                raise ValueError('AA adapter checkpoint does not match the recorded recipe')
        backbones = read(catalog / 'backbones.json')
        backbone_binding = next(b for b in backbones['bindings'] if b['recipe'] == binding_id)
        backbone = next(b for b in backbones['artifacts'] if b['sha256'] == backbone_binding['sha256'])
        if digest_file(workspace / 'clip-cache' / backbone['filename']) != backbone['sha256']:
            raise ValueError('AA backbone does not match the recorded recipe')
        # AA uses generic top-level upstream module names. Never silently reuse
        # another host's imported modules in this interpreter.
        upstream = source / 'neurips2026/AA-CLIP'
        bootstrap = next(b for b in backbones['artifacts'] if b['id'] == 'openai_vit_l14_336')
        if digest_file(upstream / 'model/ViT-L-14-336px.pt') != bootstrap['sha256']:
            raise ValueError('AA local bootstrap/model weight differs from the pinned object')
        for name in ['model', 'forward_utils', 'dataset.constants', 'utils']:
            module = sys.modules.get(name)
            filename = getattr(module, '__file__', None)
            if filename and upstream not in Path(filename).resolve().parents:
                raise RuntimeError('AA-CLIP requires an isolated host worker process')
        script = source / 'neurips2026/scripts/official_parallel_test_aaclip.py'
        spec = importlib.util.spec_from_file_location('ted_captured_aaclip_host', script)
        self.host = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.host)
        self.host._add_aaclip_to_syspath()
        self.host.ensure_aaclip_pretrained_weight()
        from model.adapter import AdaptedCLIP
        from model.clip import create_model
        from forward_utils import calculate_similarity_map, get_adapted_text_embedding
        from dataset.constants import DOMAINS
        self.device = torch.device(device)
        argv = recipe['argv']
        def integer_option(flag, default):
            return int(argv[argv.index(flag) + 1]) if flag in argv else default
        clip = create_model(model_name=summary['model_name'], img_size=summary['img_size'],
            device=self.device, pretrained=summary['pretrained'], force_image_size=summary['img_size'], require_pretrained=True)
        clip.eval()
        self.model = AdaptedCLIP(clip_model=clip, text_adapt_weight=.1, image_adapt_weight=.1,
            text_adapt_until=integer_option('--text_adapt_until', 3),
            image_adapt_until=integer_option('--image_adapt_until', 6), levels=summary['levels'], relu=False).to(self.device).eval()
        for role in ['image', 'text']:
            state = torch.load(ckpt_dir / f'{role}_adapter.pth', map_location='cpu', weights_only=True)
            self.host.load_state_project_crop(getattr(self.model, f'{role}_adapter'), state[f'{role}_adapter'], summary['adapter_load_mode'])
        self.model.requires_grad_(False)
        def captured(function):
            entries = [e for e in manifest['captured_state'] if e['function'] == function]
            if len(entries) != 1:
                raise ValueError('Missing or ambiguous captured AA state')
            entry = entries[0]
            if entry['object_path'] != 'objects/' + entry['sha256']:
                raise ValueError('Invalid captured object path')
            path = exported / entry['object_path']
            if digest_file(path) != entry['sha256']:
                raise ValueError('Captured object changed')
            return torch.load(path, map_location='cpu', weights_only=True)
        with torch.inference_mode():
            self.text = get_adapted_text_embedding(self.model, summary['target_dataset'], self.device)
            source_text = get_adapted_text_embedding(self.model, summary['source_dataset'], self.device)
            axis = self.host.mean_text_anomaly_axis(source_text)
        self.readout = CapturedAAReadout(captured('get_or_collect_source_banks'), captured('train_prescore_calibrators'),
            axis, image_size=summary['img_size'], output_mode=summary['source_prescore_calibration']['prescore_calib_output_mode'], device=self.device)
        self.transform, _ = self.host.build_transforms(summary['img_size'])
        self.similarity, self.domain = calculate_similarity_map, DOMAINS[summary['target_dataset']]
        self.summary, self.manifest = summary, manifest
        self.category = category
        if category is not None and category not in self.text:
            raise ValueError('Unknown target category')
        self.artifact_sha256 = digest_file(manifest_path)
        self._lock = threading.Lock()
        self._sync()
        self.model_load_ms = (time.perf_counter() - started) * 1000

    def _sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def info(self):
        return dict(host='AA-CLIP', recipe=self.manifest['recipe'], scope=self.manifest['scope'],
            backbone=self.manifest['backbone'], input_size=self.summary['img_size'], categories=sorted(self.text),
            category=self.category, device=str(self.device), artifact_sha256=self.artifact_sha256,
            model_load_ms=self.model_load_ms, deployment_status='experimental captured-state research bridge')

    def predict(self, image):
        if self.category is None:
            raise ValueError('AA-CLIP requires a target category')
        return self.predict_for_category(image, self.category)

    @torch.inference_mode()
    def predict_for_category(self, image, category):
        if category not in self.text or image.width * image.height > 25_000_000:
            raise ValueError('Unknown category or image exceeds pixel limit')
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Inference engine is busy')
        try:
            started = time.perf_counter()
            tensor = self.transform(image.convert('RGB')).unsqueeze(0).to(self.device)
            self._sync()
            preprocessing_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            tokens, det = self.model(tensor)
            pair = self.text[category]
            baseline = torch.stack([self.similarity(t, pair, self.summary['img_size'], test=True, domain=self.domain)[:, 0]
                                    for t in tokens], dim=1).sum(1)
            raw_image_score = ((det @ pair)[:, 1] + 1.) / 2.
            self._sync()
            host_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            corrected = self.readout(tokens, F.normalize(pair[:, 1] - pair[:, 0], dim=0), self.summary['tau'])
            self._sync()
            cted_ms = (time.perf_counter() - started) * 1000
            return dict(host_map=baseline.cpu().numpy(), cted_map=corrected.cpu().numpy(),
                image_score=float(raw_image_score[0]),
                image_score_policy='raw AA detection-token score; paper metrics additionally use dataset-wide normalization/map fusion',
                artifact_sha256=self.artifact_sha256, category=category,
                timing_ms=dict(preprocessing=preprocessing_ms, host_including_postprocess=host_ms, cted_including_postprocess=cted_ms))
        finally:
            self._lock.release()
