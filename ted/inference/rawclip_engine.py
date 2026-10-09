"""Read a passing RawCLIP captured run in an isolated CPU/GPU host worker.

No dataset loading, bank mining or calibrator fitting occurs in this bridge.
Portable acquisition and HTTP/container checks are separate release gates.
"""
import importlib
import json
from pathlib import Path
import threading
import time
import torch
import torch.nn.functional as F

from reproduction.checkpoint_download import digest_file
from .adaptclip_engine import load_relocated_module


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


class CapturedRawCLIPEngine:
    def __init__(self, *, export_directory, workspace, device='cpu', category=None):
        self.device = torch.device(device)
        if self.device.type == 'cuda':
            from reproduction.axis_run import require_live_gpu_allocation
            require_live_gpu_allocation()
        exported, workspace = Path(export_directory).resolve(), Path(workspace).resolve()
        manifest = read(exported / 'manifest.json')
        if manifest.get('schema_version') != 1 or manifest.get('host') != 'RawCLIP':
            raise ValueError('Expected a captured RawCLIP execution')
        for entry in manifest['evidence']:
            if Path(entry['file']).name != entry['file'] or digest_file(exported / entry['file']) != entry['sha256']:
                raise ValueError('Captured evaluation evidence changed')
        execution = read(exported / 'execution.json')
        if (execution.get('status') != 'matched' or execution.get('returncode') != 0
                or not execution.get('finished') or not execution['comparison']['all_match_2dp']
                or execution['recipe'] != manifest['recipe']
                or digest_file(workspace / 'run.json') != execution['plan_sha256']):
            raise ValueError('Workspace must bind a passing terminal evaluation')
        portable = workspace / 'serving-bundle.json'
        if portable.exists():
            bundle = read(portable)
            if (bundle.get('schema_version') != 1 or bundle.get('host') != 'RawCLIP'
                    or bundle.get('recipe') != manifest['recipe']
                    or bundle.get('export_sha256') != digest_file(exported / 'manifest.json')
                    or bundle.get('original_plan_sha256') != execution['plan_sha256']):
                raise ValueError('Portable RawCLIP bundle binding changed')
            for entry in bundle['files']:
                relative = Path(entry['path'])
                if (relative.is_absolute() or '..' in relative.parts
                        or digest_file(workspace / relative) != entry['sha256']
                        or (workspace / relative).stat().st_size != entry['bytes']):
                    raise ValueError('Portable RawCLIP input changed')
        source = workspace / 'source'
        if not manifest['prepared_source_files']:
            raise ValueError('Missing original evaluator source bindings')
        for entry in manifest['prepared_source_files']:
            relative = Path(entry['path'])
            if relative.is_absolute() or '..' in relative.parts or digest_file(source / relative) != entry['sha256']:
                raise ValueError('Prepared RawCLIP source changed')
        inventory = read(exported / 'capture-index.json')
        if digest_file(exported / 'capture-index.json') != execution.get('capture_index_sha256'):
            raise ValueError('Terminal captured-state inventory changed')
        states = {}
        for entry in manifest['captured_state']:
            binding = {k: entry[k] for k in ['function', 'file', 'sha256']}
            if binding not in inventory or binding not in execution['captured_files']:
                raise ValueError('Captured object differs from terminal inventory')
            if entry['object_path'] != 'objects/' + entry['sha256']:
                raise ValueError('Unexpected captured object path')
            path = exported / entry['object_path']
            if digest_file(path) != entry['sha256'] or entry['function'] in states:
                raise ValueError('Captured state changed or duplicated')
            states[entry['function']] = torch.load(path, map_location='cpu', weights_only=True)
        if set(states) != {'get_or_collect_source_patch_banks', 'train_rawclip_source_calibrators'}:
            raise ValueError('Expected exactly the original bank and calibrator captures')
        self.fp, self.defect, stats, _, metadata = states['get_or_collect_source_patch_banks']
        self.calibrators = states['train_rawclip_source_calibrators']
        self.summary = read(exported / 'summary.json')['summary']
        s = self.summary
        if stats != s['bank_stats'] and {str(k):v for k,v in stats.items()} != s['bank_stats']:
            raise ValueError('Bank statistics differ from terminal summary')
        if metadata != s['bank_cache_meta']:
            raise ValueError('Captured bank interface differs from terminal summary')
        if {str(k):v for k,v in self.calibrators.items()} != s['source_calibrators']:
            raise ValueError('Captured calibration differs from terminal summary')
        if not s['enable_calibrator']:
            raise ValueError('Expected the recorded calibrated readout')
        rows = read(exported / 'summary.json')['rows']
        self.categories = sorted({row['target_label'] for row in rows if row['target_label'] != 'mean'})
        self.category = category
        catalog = Path(__file__).resolve().parents[2] / 'reproduction'
        backbones = read(catalog / 'backbones.json')
        selected = next(b for b in backbones['bindings'] if b['recipe'] == manifest['recipe'])
        backbone = next(b for b in backbones['artifacts'] if b['sha256'] == selected['sha256'])
        weight = workspace / 'clip-cache' / backbone['filename']
        if digest_file(weight) != backbone['sha256'] or backbone['sha256'] not in {e['sha256'] for e in manifest['original_model_inputs']}:
            raise ValueError('RawCLIP backbone differs from the recorded model')
        script = source / 'neurips2026/scripts/official_parallel_test_rawclip.py'
        self.host = load_relocated_module('ted_captured_rawclip_host', script, source)
        factory = importlib.import_module('WinCLIP.CLIPAD.factory')
        openai = importlib.import_module('WinCLIP.CLIPAD.openai')
        pretrained = importlib.import_module('WinCLIP.CLIPAD.pretrained')
        expected = pretrained.get_pretrained_cfg(s['backbone'], s['pretrained_dataset'])
        original_factory, original_openai = factory.download_pretrained, openai.download_pretrained_from_url
        def pinned_config(config, *args, **kwargs):
            if config != expected:
                raise ValueError('Unexpected RawCLIP pretrained configuration')
            return str(weight)
        def pinned_url(url, *args, **kwargs):
            if url != pretrained.get_pretrained_url(s['backbone'], 'openai'):
                raise ValueError('Unexpected RawCLIP pretrained URL')
            return str(weight)
        factory.download_pretrained, openai.download_pretrained_from_url = pinned_config, pinned_url
        try:
            self.backbone = self.host.create_winclip_raw_backbone(backbone=s['backbone'],
                pretrained_dataset=s['pretrained_dataset'], device=self.device,
                img_resize=s['image_size'], img_cropsize=s['image_size'])
        finally:
            factory.download_pretrained, openai.download_pretrained_from_url = original_factory, original_openai
        self.transform, _ = self.host.build_transforms(s['image_size'])
        self.artifact_sha256 = digest_file(exported / 'manifest.json')
        self._lock = threading.Lock()

    def info(self):
        return dict(host='RawCLIP', backbone=self.summary['backbone'], device=str(self.device),
            image_size=self.summary['image_size'], category=self.category, categories=self.categories,
            artifact_sha256=self.artifact_sha256, readouts=['baseline_txt','parallel_margin','ted_calibrated'],
            state='terminal captured source bank/calibrators; no inference-time fitting')

    def predict_for_category(self, image, category):
        return self.predict(image, category=category)

    @torch.inference_mode()
    def predict(self, image, category=None):
        category = category or self.category
        if category not in self.categories:
            raise ValueError('An explicit recorded target category is required')
        if image.width * image.height > 25_000_000:
            raise ValueError('Image exceeds pixel limit')
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Inference engine is busy')
        try:
            start = time.perf_counter()
            s, h = self.summary, self.host
            tensor = self.transform(image.convert('RGB')).unsqueeze(0).to(self.device)
            pair = h.encode_prompt_pair(self.backbone, category, s['prompt_mode'])
            feature = h.encode_global_image_feature(self.backbone, tensor)
            logits = feature @ torch.stack([pair['norm_img'], pair['anom_img']], dim=1)
            global_score = (logits / .07).softmax(dim=-1)[0, 1].item()
            patches = self.backbone.extract_layer_patch_tokens(tensor, target_layers=s['features_list'])
            baseline_layers, parallel_layers, residual_layers = [], [], []
            for index, layer in enumerate(s['features_list']):
                probs, margin = h.compute_layer_host_and_ted(patches[layer][0], pair,
                    self.fp[index], self.defect[index], tau=s['tau'], fp_weight=s['fp_weight'])
                height, width = self.backbone.grid_size
                if probs.shape[0] != height * width:
                    height = int(round(probs.shape[0] ** .5))
                    width = probs.shape[0] // height
                size = (s['image_size'], s['image_size'])
                baseline_layers.append(F.interpolate(probs.reshape(1,1,height,width), size=size,
                    mode='bilinear', align_corners=False)[:,0])
                parallel = F.interpolate(margin.reshape(1,1,height,width), size=size,
                    mode='bilinear', align_corners=False)[:,0]
                parallel_layers.append(parallel)
                residual_layers.append(float(self.calibrators.get(index, {}).get('eta', 0.)) * h.zscore_tensor(parallel))
            baseline = h.smooth_map(torch.stack(baseline_layers).mean(dim=0), sigma=s['sigma'])
            parallel = h.smooth_map(torch.stack(parallel_layers).mean(dim=0), sigma=s['sigma'])
            parallel = h.activate_margin_map(parallel, activation=s['map_activation'], temperature=s['map_temperature'])
            if s['hybrid_source'] != 'none' and s['hybrid_weight'] > 0:
                if s['hybrid_source'] != 'baseline':
                    raise ValueError('Unsupported recorded hybrid source')
                parallel = (1-s['hybrid_weight']) * parallel + s['hybrid_weight'] * baseline
            residual = h.smooth_map(torch.stack(residual_layers).mean(dim=0), sigma=s['sigma'])
            anchor = baseline if s['calib_anchor'] == 'baseline' else parallel
            corrected = anchor + s['calib_residual_scale'] * anchor.std().clamp_min(1e-6) * residual
            def score(value):
                return s['image_fusion_weight'] * global_score + (1-s['image_fusion_weight']) * h.topk_mean(value[0].numpy(), s['image_score_topk'])
            return dict(host_map=baseline, tted_map=parallel, cted_map=corrected,
                image_score=score(baseline), tted_image_score=score(parallel), cted_image_score=score(corrected),
                image_score_policy='recorded global prompt score and top-k map fusion for each readout',
                category=category, artifact_sha256=self.artifact_sha256,
                timing_ms=dict(total=(time.perf_counter()-start)*1000))
        finally:
            self._lock.release()
