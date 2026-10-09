import json
from pathlib import Path
import unittest

from reproduction.checkpoint_download import digest_file
from reproduction.metrics import compare, extract, extract_recipe
from reproduction.recipe_lookup import reference_summary, execution_recipe


ROOT = Path(__file__).resolve().parents[1] / 'reproduction'


class PublishedEvidenceTests(unittest.TestCase):
    def test_rawclip_anonymous_acquisition_and_image_readouts_are_verified(self):
        folder = ROOT / 'validation/a10-20261009/rawclip-public-client-v1'
        index = json.loads((folder / 'index.json').read_text())
        self.assertTrue(index['anonymous_acquisition_verified'])
        self.assertTrue(index['anonymous_three_image_parity_verified'])
        for item in index['files']:
            self.assertEqual(digest_file(folder / item['file']), item['sha256'])
        acquisition = json.loads((folder / 'rawclip-openai-anonymous-acquisition-20261009-v1.json').read_text())
        self.assertEqual(acquisition['files'], 930)
        self.assertEqual(acquisition['status'], 'verified_serving_inputs')
        parity = json.loads((folder / 'rawclip-openai-anonymous-image-parity-20261009-v1.json').read_text())
        self.assertEqual(parity['status'], 'matched')
        self.assertTrue(parity['network_denied'])
        self.assertTrue(parity['original_workspace_reads_denied'])
        self.assertFalse(parity['gpu_used'])
        self.assertEqual(len(parity['cases']), 3)
        for case in parity['cases']:
            self.assertEqual(set(case['map_max_abs_error']), {'host_map','tted_map','cted_map'})
            self.assertFalse(any(case['map_max_abs_error'].values()))
            self.assertFalse(any(case['score_abs_error'].values()))

    def test_rawclip_public_catalog_binds_verified_metadata_to_immutable_revision(self):
        folder = ROOT / 'validation/a10-20261009/rawclip-public-v1'
        index = json.loads((folder / 'index.json').read_text())
        catalog = json.loads((ROOT / 'rawclip-serving-releases.json').read_text())
        record = catalog['releases']['openai-l14']
        self.assertEqual(record['host'], 'RawCLIP')
        self.assertEqual(record['revision'], index['revision'])
        self.assertEqual(record['files'], 930)
        self.assertFalse(index['whole_paper_reproduced'])
        for item in index['files']:
            self.assertEqual(digest_file(folder / item['file']), item['sha256'])
            self.assertIn('/resolve/' + record['revision'] + '/', item['url'])
        original = json.loads((folder / 'rawclip-openai-archive-20261009-v1.json').read_text())
        self.assertEqual(original, {k:v for k,v in record.items() if k != 'revision'})

    def test_eight_worker_gateway_preserves_all_readouts_and_recorded_strengths(self):
        folder = ROOT / 'validation/a10-20261009/gateway-v3'
        index = json.loads((folder / 'index.json').read_text())
        self.assertFalse(index['whole_paper_reproduced'])
        for item in index['files']:
            self.assertEqual(digest_file(folder / item['file']), item['sha256'])
        self.assertEqual(digest_file(folder / 'proof.zip'), index['archive_sha256'])
        report = json.loads((folder / 'gateway.json').read_text())
        self.assertEqual(report['status'], 'matched')
        self.assertEqual((report['models'], report['cases'], len(report['rows'])), (8, 36, 36))
        self.assertEqual(digest_file(folder / 'models.json'), report['registry_sha256'])
        self.assertEqual(report['unknown_model_status'], 404)
        self.assertEqual(report['missing_aa_category_status'], 422)
        raw = [r for r in report['rows'] if r['model'] == 'rawclip-openai-main']
        self.assertEqual(len(raw), 3)
        for row in report['rows']:
            for key in ['host_max_abs_error', 'cted_max_abs_error', 'raw_image_score_abs_error']:
                self.assertEqual(row[key], 0)
        for row in raw:
            self.assertEqual(row['tted_max_abs_error'], 0)
            self.assertEqual(row['tted_image_score_abs_error'], 0)

    def test_public_updated_model_card_and_proofs_match_immutable_hf_bytes(self):
        folder = ROOT / 'validation/a10-20261009/hf-card-v1'
        index = json.loads((folder / 'index.json').read_text())
        self.assertFalse(index['whole_paper_reproduced'])
        self.assertEqual(index['manifest_entries'],90)
        for item in index['files']:
            self.assertEqual(digest_file(folder / item['file']),item['sha256'])
            self.assertEqual(item['authentication'],'none')
            self.assertIn('/resolve/' + index['revision'] + '/',item['url'])
        for key,source in [('bplus','faprompt-public-client-v1'),('h14','faprompt-public-client-v2')]:
            for kind,name in [('acquisition','anonymous-acquisition'),('image','anonymous-image-parity')]:
                self.assertEqual((folder / ('faprompt-' + key + '-' + name + '.json')).read_bytes(),
                                 (folder.parent / source / (key + '-' + kind + '.json')).read_bytes())
        self.assertEqual((folder / 'seven-worker-gateway-parity.json').read_bytes(),
                         (folder.parent / 'gateway-v2/gateway.json').read_bytes())

    def test_h14_anonymous_acquisition_and_every_image_strength_are_verified(self):
        import zipfile
        folder = ROOT / 'validation/a10-20261009/faprompt-public-client-v2'
        index = json.loads((folder / 'index.json').read_text())
        self.assertTrue(index['h14_anonymous_acquisition_verified'])
        self.assertTrue(index['h14_image_parity_verified'])
        self.assertFalse(index['whole_paper_reproduced'])
        self.assertEqual(digest_file(folder / 'proof.zip'),index['archive_sha256'])
        self.assertEqual(digest_file(folder / index['bplus_evidence']['path']),index['bplus_evidence']['sha256'])
        with zipfile.ZipFile(folder / 'proof.zip') as archive:
            for item in index['files']:
                self.assertEqual(digest_file(folder / item['file']),item['sha256'])
                self.assertEqual((folder / item['file']).read_bytes(),archive.read(item['archive_member']))
        acquisition = json.loads((folder / 'h14-acquisition.json').read_text())
        self.assertEqual(acquisition['status'],'verified_serving_inputs')
        self.assertEqual(acquisition['files'],932)
        self.assertEqual(acquisition['revision'],index['revision'])
        self.assertEqual(acquisition['authentication'],'none; standard-library HTTPS without auth headers')
        image = json.loads((folder / 'h14-image.json').read_text())
        self.assertEqual(image['status'],'matched')
        self.assertEqual({(r['category'],r['alpha']) for r in image['rows']},
                         {(c,a) for c in ['01','02','03'] for a in [.5,1.,1.5]})
        for flag in ['original_input_reads_denied','denial_self_checks_passed','network_denied']:
            self.assertTrue(image[flag])
        self.assertFalse(image['fitting_performed'])
        for row in image['rows']:
            for key in ['host_max_absolute_error','cted_max_absolute_error','host_image_score_error','cted_image_score_error']:
                self.assertEqual(row[key],0)

    def test_bplus_anonymous_public_client_and_images_are_independently_verified(self):
        folder = ROOT / 'validation/a10-20261009/faprompt-public-client-v1'
        index = json.loads((folder / 'index.json').read_text())
        self.assertTrue(index['bplus_anonymous_acquisition_verified'])
        self.assertTrue(index['bplus_image_parity_verified'])
        self.assertFalse(index['h14_anonymous_image_parity_verified'])
        self.assertFalse(index['whole_paper_reproduced'])
        for item in index['files']:
            self.assertEqual(digest_file(folder / item['file']),item['sha256'])
        acquisition = json.loads((folder / 'bplus-acquisition.json').read_text())
        self.assertEqual(acquisition['status'],'verified_serving_inputs')
        self.assertEqual(acquisition['files'],932)
        self.assertEqual(acquisition['revision'],index['revision'])
        self.assertEqual(acquisition['authentication'],'none; standard-library HTTPS without auth headers')
        image = json.loads((folder / 'bplus-image.json').read_text())
        self.assertEqual(image['status'],'matched')
        self.assertEqual({(r['category'],r['alpha']) for r in image['rows']},
                         {(c,a) for c in ['01','02','03'] for a in [.5,1.,1.5]})
        self.assertTrue(image['original_input_reads_denied'])
        self.assertTrue(image['denial_self_checks_passed'])
        self.assertTrue(image['network_denied'])
        for row in image['rows']:
            for key in ['host_max_absolute_error','cted_max_absolute_error','host_image_score_error','cted_image_score_error']:
                self.assertEqual(row[key],0)
        folder = ROOT / 'validation/a10-20261009/rawclip-openai-capture-v1'
        index = json.loads((folder / 'index.json').read_text())
        for item in index['files']:
            self.assertEqual(digest_file(folder / item['file']),item['sha256'])
        self.assertFalse(index['image_inference_verified'])
        manifest = json.loads((folder / 'openai/manifest.json').read_text())
        self.assertEqual(manifest['capture_binding'],'terminal_execution_record')
        self.assertEqual(len(manifest['captured_state']),2)

    def test_faprompt_public_catalog_is_bound_to_anonymous_metadata_and_saved_archives(self):
        folder = ROOT / 'validation/a10-20261009/faprompt-public-v1'
        metadata = json.loads((folder / 'metadata.json').read_text())
        self.assertFalse(metadata['archives_fully_downloaded_and_verified'])
        self.assertEqual(len(metadata['files']),4)
        for item in metadata['files']:
            self.assertEqual(digest_file(folder / item['file']),item['sha256'])
            self.assertEqual(item['authentication'],'none')
            self.assertIn('/resolve/' + metadata['revision'] + '/',item['url'])
        catalog = json.loads((ROOT / 'faprompt-serving-releases.json').read_text())
        self.assertEqual(catalog['repository'],'KIMJINYOUNG/TED-reproducibility')
        for key,variant in [('bplus-seed1','bplus'),('h14-seed0','h14')]:
            published = dict(catalog['releases'][key])
            self.assertEqual(published.pop('revision'),metadata['revision'])
            archived = json.loads((ROOT / ('validation/a10-20261009/faprompt-images-v2/' + variant + '-archive.json')).read_text())
            self.assertEqual(published,archived)
            self.assertEqual(published['files'],932)
            self.assertEqual(published['host'],'FAPrompt')
        browser = ROOT / 'validation/a10-20261009/gateway-v2'
        record = json.loads((browser / 'index.json').read_text())['browser_validation']
        self.assertEqual(digest_file(browser / record['file']),record['sha256'])
        text = (browser / record['file']).read_text(encoding='utf-8')
        self.assertIn('Inspection complete.',text)
        self.assertIn('C-TED strength: 1.5',text)
        self.assertIn('Release: faprompt-h14-captured',text)

    def test_seven_worker_gateway_proof_covers_every_recorded_strength(self):
        import zipfile
        folder = ROOT / 'validation/a10-20261009/gateway-v2'
        index = json.loads((folder / 'index.json').read_text())
        self.assertFalse(index['standalone_image_verified'])
        self.assertFalse(index['whole_paper_reproduced'])
        self.assertEqual(digest_file(folder / 'proof.zip'), index['archive_sha256'])
        with zipfile.ZipFile(folder / 'proof.zip') as archive:
            for item in index['files']:
                self.assertEqual(digest_file(folder / item['file']), item['sha256'])
                self.assertEqual((folder / item['file']).read_bytes(), archive.read(item['archive_member']))
        registry = json.loads((folder / 'models.json').read_text())['models']
        report = json.loads((folder / 'gateway.json').read_text())
        self.assertEqual(report['registry_sha256'], digest_file(folder / 'models.json'))
        self.assertEqual(report['status'], 'matched')
        self.assertEqual(report['cases'], 33)
        self.assertEqual(report['models'], 7)
        expected = {(m['id'],c,a) for m in registry for c in ['01','02','03']
                    for a in ([.5,1.,1.5] if m['id'].endswith('-captured') else [None])}
        self.assertEqual({(r['model'],r['fixture_category'],r.get('alpha')) for r in report['rows']}, expected)
        artifacts = {m['id']:m['artifact_sha256'] for m in registry}
        for row in report['rows']:
            self.assertEqual(row['artifact_sha256'],artifacts[row['model']])
            for key in ['host_max_abs_error','cted_max_abs_error','raw_image_score_abs_error']:
                self.assertEqual(row[key],0)
            self.assertEqual(row.get('cted_image_score_abs_error',0),0)
        self.assertEqual(report['unknown_model_status'],404)
        self.assertEqual(report['missing_aa_category_status'],422)

    def test_rawclip_captured_exports_retain_terminal_bindings_without_serving_claim(self):
        folder = ROOT / 'validation/a10-20261009/rawclip-captures-v1'
        index = json.loads((folder / 'index.json').read_text())
        for flag in ['image_inference_verified','hf_publication_verified','docker_verified','whole_paper_reproduced']:
            self.assertFalse(index[flag])
        for item in index['files']:
            self.assertEqual(digest_file(folder / item['file']), item['sha256'])
        for key in ['h14','l336']:
            manifest = json.loads((folder / key / 'manifest.json').read_text())
            execution = json.loads((folder / key / 'execution.json').read_text())
            self.assertEqual(manifest['capture_binding'], 'terminal_execution_record')
            self.assertEqual(execution['status'], 'matched')
            self.assertEqual(len(manifest['captured_state']), 2)
            self.assertEqual([r['sha256'] for r in manifest['captured_state']],
                             [r['sha256'] for r in execution['captured_files']])
            for item in manifest['evidence']:
                self.assertEqual(digest_file(folder / key / item['file']), item['sha256'])

    def test_faprompt_docker_matches_all_canonical_images_and_recorded_strengths(self):
        import zipfile
        folder = ROOT / 'validation/a10-20261009/faprompt-images-v3'
        index = json.loads((folder / 'index.json').read_text())
        self.assertTrue(index['docker_http_verified'])
        self.assertFalse(index['hf_publication_verified'])
        self.assertFalse(index['whole_paper_reproduced'])
        self.assertEqual(digest_file(folder / 'proof.zip'), index['archive_sha256'])
        with zipfile.ZipFile(folder / 'proof.zip') as archive:
            for item in index['files']:
                self.assertEqual(digest_file(folder / item['file']), item['sha256'])
                self.assertEqual((folder / item['file']).read_bytes(), archive.read(item['archive_member']))
        for key, port in [('bplus',18088), ('h14',18089)]:
            report = json.loads((folder / (key + '-http.json')).read_text())
            self.assertEqual(report['status'], 'matched')
            self.assertTrue(report['http_verified'])
            self.assertTrue(report['original_input_reads_denied'])
            self.assertTrue(report['denial_self_checks_passed'])
            self.assertFalse(report['fitting_performed'])
            self.assertEqual(report['allowed_loopback_http'], f'http://127.0.0.1:{port}')
            self.assertEqual({(r['category'],r['alpha']) for r in report['rows']},
                             {(c,a) for c in ['01','02','03'] for a in [.5,1.,1.5]})
            for row in report['rows']:
                for metric in ['host_max_absolute_error','cted_max_absolute_error',
                               'host_image_score_error','cted_image_score_error']:
                    self.assertEqual(row[metric], 0)
            container = json.loads((folder / (key + '-container.json')).read_text())
            self.assertEqual(container['health'], 'healthy')
            self.assertTrue(container['read_only'])
            self.assertEqual(container['ports']['8000/tcp'][0]['HostIp'], '127.0.0.1')

    def test_faprompt_seeded_images_and_archive_checks_cover_every_strength(self):
        import zipfile
        folder = ROOT / 'validation/a10-20261009/faprompt-images-v2'
        index = json.loads((folder / 'index.json').read_text())
        self.assertFalse(index['docker_http_verified'])
        self.assertFalse(index['hf_publication_verified'])
        self.assertFalse(index['whole_paper_reproduced'])
        self.assertEqual(digest_file(folder / 'proof.zip'), index['archive_sha256'])
        with zipfile.ZipFile(folder / 'proof.zip') as archive:
            for item in index['files']:
                path = folder / item['file']
                self.assertEqual(digest_file(path), item['sha256'])
                self.assertEqual(path.read_bytes(), archive.read(item['archive_member']))
        for name in ['h14-image.json','bplus-fresh-image.json','bplus-parameter.json']:
            report = json.loads((folder / name).read_text())
            self.assertEqual(report['status'], 'matched')
            self.assertTrue(report['denial_self_checks_passed'])
            self.assertEqual({(r['category'], r['alpha']) for r in report['rows']},
                             {(c,a) for c in ['01','02','03'] for a in [.5,1.,1.5]})
            for row in report['rows']:
                for key in ['host_max_absolute_error','cted_max_absolute_error',
                            'host_image_score_error','cted_image_score_error']:
                    self.assertEqual(row[key], 0)
        acquisition = json.loads((folder / 'bplus-unpack.json').read_text())
        record = json.loads((folder / 'bplus-archive.json').read_text())
        self.assertEqual(acquisition['status'], 'verified_serving_inputs')
        self.assertEqual(acquisition['archive_sha256'], record['sha256'])
        self.assertEqual(acquisition['files'], 932)

    def test_faprompt_bplus_images_match_and_h14_failure_is_retained(self):
        import zipfile
        folder = ROOT / 'validation/a10-20261009/faprompt-images-v1'
        index = json.loads((folder / 'index.json').read_text())
        self.assertFalse(index['docker_http_verified'])
        self.assertFalse(index['hf_publication_verified'])
        self.assertEqual(digest_file(folder / 'proof.zip'), index['archive_sha256'])
        with zipfile.ZipFile(folder / 'proof.zip') as archive:
            for item in index['files']:
                path = folder / item['file']
                self.assertEqual(digest_file(path), item['sha256'])
                self.assertEqual(path.read_bytes(), archive.read(item['archive_member']))
        report = json.loads((folder / 'bplus-image-parity-v3.json').read_text())
        self.assertEqual(report['status'], 'matched')
        self.assertTrue(report['denial_self_checks_passed'])
        self.assertTrue(report['original_input_reads_denied'])
        self.assertTrue(report['network_denied'])
        self.assertFalse(report['fitting_performed'])
        self.assertEqual({(r['category'], r['alpha']) for r in report['rows']},
                         {(c, a) for c in ['01','02','03'] for a in [.5,1.,1.5]})
        for row in report['rows']:
            for key in ['host_max_absolute_error','cted_max_absolute_error',
                        'host_image_score_error','cted_image_score_error']:
                self.assertEqual(row[key], 0)
        failed = json.loads((folder / 'h14-image-parity-v1.json').read_text())
        self.assertEqual(failed['status'], 'failed')
        self.assertIn('Tensor-likes are not close', failed['error'])

    def test_faprompt_packaging_retains_passing_runs_and_branch_identity(self):
        folder = ROOT / 'validation/a10-20261009/faprompt-package-v1'
        index = json.loads((folder / 'index.json').read_text())
        self.assertFalse(index['inference_parity_verified'])
        self.assertFalse(index['docker_http_verified'])
        self.assertFalse(index['hf_publication_verified'])
        for variant in ['h14', 'bplus']:
            readout = json.loads((folder / (variant + '-readout.json')).read_text())
            self.assertEqual(readout['status'], 'matched_within_declared_tolerance')
            self.assertFalse(readout['image_inference_verified'])
            self.assertFalse(readout['fitting_performed'])
            summary = json.loads((folder / variant / 'summary.json').read_text())
            self.assertEqual([r['alpha'] for r in readout['rows']], summary['alphas'])
            for row in readout['rows']:
                self.assertEqual(row['max_absolute_error'], 0)
        for item in index['files']:
            self.assertEqual(digest_file(folder / item['file']), item['sha256'])
        for variant in ['h14', 'bplus']:
            bundle = json.loads((folder / (variant + '-bundle.json')).read_text())
            manifest = folder / variant / 'manifest.json'
            self.assertEqual(digest_file(manifest), bundle['export_sha256'])
            self.assertFalse(bundle['fitting_performed'])
            self.assertEqual(len(bundle['files']), 923)
            execution = json.loads((folder / variant / 'execution.json').read_text())
            self.assertEqual(execution['status'], 'matched')
            self.assertTrue(execution['comparison']['all_match_2dp'])
            branches = json.loads((folder / (variant + '-branches.json')).read_text())
            self.assertEqual(branches['export_sha256'], bundle['export_sha256'])
            self.assertFalse(branches['inference_parity_verified'])
            self.assertEqual({b['branch'] for b in branches['branches']}, {'branch1', 'branch2'})
            summary = json.loads((folder / variant / 'summary.json').read_text())
            for branch in branches['branches']:
                self.assertTrue(branch['basis_finite'])
                self.assertEqual(branch['basis_shape'][1], 4)
                self.assertEqual(branch['summary_metadata'],
                                 summary['source_branch_calibrators'][branch['branch']])

    def test_five_worker_docker_evidence_preserves_maps_and_corrected_scores(self):
        import zipfile
        folder = ROOT / 'validation/a10-20261009/docker-v1'
        index = json.loads((folder / 'index.json').read_text())
        self.assertFalse(index['standalone_image_verified'])
        self.assertFalse(index['whole_paper_reproduced'])
        self.assertEqual(digest_file(ROOT / index['archive']['path']), index['archive']['sha256'])
        with zipfile.ZipFile(ROOT / index['archive']['path']) as archive:
            for item in index['files']:
                path = ROOT / item['path']
                self.assertEqual(digest_file(path), item['sha256'])
                self.assertEqual(path.read_bytes(), archive.read(item['archive_member']))
        # This proof binds the historical five-worker snapshot, not future registries.
        self.assertEqual(len(json.loads((folder / 'models.json').read_text())['models']), 5)
        report = json.loads((folder / 'gateway-v2.json').read_text())
        self.assertEqual(report['status'], 'matched')
        self.assertEqual(report['cases'], 15)
        self.assertEqual(report['registry_sha256'], digest_file(folder / 'models.json'))
        registry = json.loads((folder / 'models.json').read_text())['models']
        releases = {m['id']: m['artifact_sha256'] for m in registry}
        self.assertEqual({(r['model'], r['fixture_category']) for r in report['rows']},
                         {(m, c) for m in releases for c in ['01', '02', '03']})
        for row in report['rows']:
            self.assertEqual(row['artifact_sha256'], releases[row['model']])
            for key in ['host_max_abs_error', 'cted_max_abs_error', 'raw_image_score_abs_error']:
                self.assertEqual(row[key], 0)
            if row['model'].startswith('adaptclip-'):
                self.assertEqual(row['cted_image_score_abs_error'], 0)
        self.assertEqual(report['unknown_model_status'], 404)
        self.assertEqual(report['missing_aa_category_status'], 422)
        for variant in ['openai', 'l336']:
            direct = json.loads((folder / (variant + '-http.json')).read_text())
            self.assertEqual(direct['status'], 'matched')
            self.assertEqual({r['category'] for r in direct['rows']}, {'01', '02', '03'})
            for row in direct['rows']:
                for key in ['host_max_absolute_error', 'cted_max_absolute_error',
                            'host_image_score_error', 'cted_image_score_error']:
                    self.assertEqual(row[key], 0)

    def test_pilab_adaptclip_cpu_proof_does_not_claim_container_verification(self):
        import zipfile
        index=json.loads((ROOT/'adaptclip-serving-exports.json').read_text())
        proof=index['pilab_cpu_validation']
        self.assertFalse(proof['deployment_ready'])
        self.assertFalse(proof['docker_http_verified'])
        self.assertFalse(proof['standalone_image_verified'])
        archive=ROOT/proof['archive']['path']
        self.assertEqual(digest_file(archive),proof['archive']['sha256'])
        with zipfile.ZipFile(archive) as bundle:
            for entry in proof['files']:
                path=ROOT/entry['path']
                self.assertEqual(digest_file(path),entry['sha256'])
                self.assertEqual(path.read_bytes(),bundle.read(entry['archive_member']))
        catalog=json.loads((ROOT/'adaptclip-serving-releases.json').read_text())
        for variant in ['openai','l336']:
            acquisition=json.loads((archive.parent/(variant+'-acquisition.json')).read_text())
            parity=json.loads((archive.parent/(variant+'-parity.json')).read_text())
            record=catalog['releases'][variant+'-seed0']
            self.assertEqual(acquisition['status'],'verified_serving_inputs')
            self.assertEqual(acquisition['archive_sha256'],record['sha256'])
            self.assertEqual(parity['engine']['artifact_sha256'],record['export_sha256'])
            self.assertEqual(parity['status'],'matched')
            self.assertEqual(parity['device'],'cpu')
            self.assertTrue(parity['denial_self_checks_passed'])
            self.assertFalse(parity['fixture_metadata_read_exception'])
            self.assertEqual(len(parity['rows']),3)
            for row in parity['rows']:
                for key in ['host_max_absolute_error','cted_max_absolute_error',
                            'host_image_score_error','cted_image_score_error']:
                    self.assertEqual(row[key],0)

    def test_public_adaptclip_acquisition_and_inference_bind_pinned_archives(self):
        import zipfile
        index=json.loads((ROOT/'adaptclip-serving-exports.json').read_text())
        catalog=json.loads((ROOT/'adaptclip-serving-releases.json').read_text())
        proof=index['anonymous_validation']
        self.assertTrue(proof['anonymous_acquisition_and_image_inference_verified'])
        self.assertFalse(proof['deployment_ready'])
        self.assertFalse(proof['docker_http_verified'])
        archive=ROOT/proof['archive']['path']
        self.assertEqual(digest_file(archive),proof['archive']['sha256'])
        with zipfile.ZipFile(archive) as bundle:
            for entry in proof['files']:
                path=ROOT/entry['path']
                self.assertEqual(digest_file(path),entry['sha256'])
                if path.name!='public-metadata-verification.json':
                    self.assertEqual(path.read_bytes(),bundle.read(entry.get('archive_member',path.name)))
        for variant in ['openai','l336']:
            record=catalog['releases'][variant+'-seed0']
            acquired=json.loads((archive.parent/(variant+'-acquisition.json')).read_text())
            parity=json.loads((archive.parent/(variant+'-parity.json')).read_text())
            self.assertEqual(acquired['status'],'verified_serving_inputs')
            self.assertEqual(acquired['files'],931)
            self.assertEqual(acquired['archive_sha256'],record['sha256'])
            self.assertEqual(acquired['revision'],record['revision'])
            self.assertEqual(acquired['revision'],proof['revision'])
            self.assertEqual(acquired['authentication'],'none; standard-library HTTPS without auth headers')
            self.assertEqual(parity['status'],'matched')
            self.assertEqual(parity['engine']['artifact_sha256'],record['export_sha256'])
            self.assertTrue(parity['portable_bundle_verified'])
            self.assertTrue(parity['denial_self_checks_passed'])
            self.assertFalse(parity['fixture_metadata_read_exception'])
            original=json.loads((ROOT/'validation/a10-20261009'/('adaptclip-'+variant+'-image-parity-20261009-v1.json')).read_text())
            self.assertEqual([(r['image'],r['image_sha256']) for r in parity['rows']],
                             [(r['image'],r['image_sha256']) for r in original['rows']])
            for row in parity['rows']:
                for key in ['host_max_absolute_error','cted_max_absolute_error',
                            'host_image_score_error','cted_image_score_error']:
                    self.assertEqual(row[key],0)

    def test_adaptclip_fresh_archive_parity_binds_every_proof_file(self):
        import zipfile
        index = json.loads((ROOT / 'adaptclip-serving-exports.json').read_text())
        proof = index['portable_validation']
        self.assertFalse(proof['deployment_ready'])
        self.assertFalse(proof['anonymous_download_verified'])
        self.assertFalse(proof['docker_http_verified'])
        archive = ROOT / proof['archive']['path']
        self.assertEqual(digest_file(archive), proof['archive']['sha256'])
        with zipfile.ZipFile(archive) as source:
            for binding in proof['files']:
                path = ROOT / binding['path']
                self.assertEqual(path.stat().st_size, binding['bytes'])
                self.assertEqual(digest_file(path), binding['sha256'])
                self.assertEqual(source.read(path.name), path.read_bytes())
        folder = archive.parent
        for variant in ['openai','l336']:
            report = json.loads((folder / ('adaptclip-'+variant+'-unpacked-parity-20261009-v1.json')).read_text())
            record = json.loads((folder / ('adaptclip-'+variant+'-archive-20261009-v1.json')).read_text())
            unpacked = json.loads((folder / ('adaptclip-'+variant+'-unpack-20261009-v1.json')).read_text())
            manifest = json.loads((folder / (variant+'-serving-bundle.json')).read_text())
            self.assertEqual(record['files'],931)
            self.assertEqual(record['host'],'AdaptCLIP')
            self.assertEqual(record['sha256'],unpacked['archive_sha256'])
            self.assertEqual(record['export_sha256'],report['engine']['artifact_sha256'])
            self.assertEqual(record['export_sha256'],manifest['export_sha256'])
            self.assertEqual(sum(e['path'].startswith('source/') for e in manifest['files']),920)
            self.assertEqual(report['status'],'matched')
            self.assertTrue(report['engine']['portable_bundle'])
            self.assertTrue(report['portable_bundle_verified'])
            self.assertTrue(report['original_workspace_model_reads_and_network_denied'])
            self.assertTrue(report['fixture_metadata_read_exception'])
            self.assertFalse(report['model_fitting'])
            self.assertFalse(report['anonymous_download_verified'])
            self.assertFalse(report['docker_http_verified'])
            self.assertEqual({r['category'] for r in report['rows']},{'01','02','03'})
            for row in report['rows']:
                for key in ['host_max_absolute_error','cted_max_absolute_error',
                            'host_image_score_error','cted_image_score_error']:
                    self.assertEqual(row[key],0)

    def test_adaptclip_real_image_parity_keeps_portable_and_docker_gates(self):
        import zipfile
        index = json.loads((ROOT / 'adaptclip-serving-exports.json').read_text())
        proof = index['image_parity']
        archive = ROOT / proof['archive']['path']
        self.assertEqual(digest_file(archive), proof['archive']['sha256'])
        observed = set()
        with zipfile.ZipFile(archive) as bundle:
            for item in proof['reports']:
                path = ROOT / item['path']
                self.assertEqual(digest_file(path), item['sha256'])
                self.assertEqual(bundle.read(path.name), path.read_bytes())
                report = json.loads(path.read_text())
                self.assertEqual(report['status'], 'matched')
                self.assertEqual(report['device'], 'cpu')
                self.assertFalse(report['model_fitting'])
                self.assertFalse(report['portable_bundle_verified'])
                self.assertFalse(report['docker_http_verified'])
                observed.add(report['engine']['recipe'])
                self.assertEqual({r['category'] for r in report['rows']}, {'01','02','03'})
                for row in report['rows']:
                    self.assertEqual(row['map_shape'], [1,518,518])
                    for key in ['host_max_absolute_error', 'cted_max_absolute_error',
                                'host_image_score_error', 'cted_image_score_error']:
                        self.assertEqual(row[key], 0)
        self.assertEqual(observed, {entry['recipe'] for entry in index['exports']})

    def test_adaptclip_exports_bind_terminal_capture_and_retain_adapter_gate(self):
        import hashlib
        import zipfile
        index = json.loads((ROOT / 'adaptclip-serving-exports.json').read_text())
        binding = index['readout_validation']
        raw = (ROOT / binding['path']).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), binding['sha256'])
        proof = json.loads(raw)
        self.assertEqual(proof['tests'], 2)
        self.assertEqual(proof['device'], 'cpu')
        self.assertFalse(proof['full_image_inference_verified'])
        self.assertFalse(proof['deployment_ready'])
        self.assertEqual(digest_file(ROOT / binding['archive']['path']), binding['archive']['sha256'])
        with zipfile.ZipFile(ROOT / binding['archive']['path']) as archive:
            self.assertEqual(archive.read('validation.json'), raw)
            log = archive.read(proof['log']['file'])
            self.assertEqual(hashlib.sha256(log).hexdigest(), proof['log']['sha256'])
        self.assertEqual(len(index['exports']), 2)
        for entry in index['exports']:
            self.assertFalse(entry['deployment_ready'])
            binding = entry['archive']
            path = ROOT / binding['path']
            self.assertEqual(path.stat().st_size, binding['bytes'])
            self.assertEqual(digest_file(path), binding['sha256'])
            with zipfile.ZipFile(path) as bundle:
                manifest = json.loads(bundle.read('manifest.json'))
                execution = json.loads(bundle.read('execution.json'))
                self.assertEqual(manifest['recipe'], entry['recipe'])
                self.assertEqual(manifest['capture_binding'], 'terminal_execution_record')
                self.assertEqual(execution['status'], 'matched')
                self.assertEqual(len(manifest['prepared_source_files']), 920)
                self.assertEqual(manifest['captured_state'], entry['calibrators'])
                self.assertEqual(len(manifest['captured_state']), 2)
                for item in manifest['captured_state']:
                    raw = bundle.read(item['object_path'])
                    self.assertEqual(hashlib.sha256(raw).hexdigest(), item['sha256'])
                    self.assertEqual(len(raw), item['bytes'])
                for item in manifest['evidence']:
                    self.assertEqual(hashlib.sha256(bundle.read(item['file'])).hexdigest(), item['sha256'])

    def test_residual_btad_fresh_replay_preserves_all_44_comparisons(self):
        self.check_residual_replay('btad', 36, 741)

    def test_residual_mpdd_fresh_replay_preserves_all_44_comparisons(self):
        self.check_residual_replay('mpdd', 31, 458)

    def test_residual_visa_fresh_replay_preserves_all_44_comparisons(self):
        self.check_residual_replay('visa', 35, 2162)

    def check_residual_replay(self, dataset, matched, images):
        folder = ROOT / ('validation/a10-20261009/residual-' + dataset + '-v5')
        execution = json.loads((folder / 'execution.json').read_text())
        comparison = json.loads((folder / 'comparison.json').read_text())
        actual = json.loads((folder / 'summary.json').read_text())
        recipe = execution_recipe(ROOT, execution['recipe'])
        reference, expected = reference_summary(ROOT, execution['recipe'])
        cells = compare(extract_recipe(actual, recipe), extract_recipe(expected, recipe))
        self.assertEqual(comparison, execution['comparison'])
        self.assertEqual(cells, comparison['cells'])
        self.assertEqual(len(cells), 44)
        self.assertEqual(sum(c['matches_printed_precision'] for c in cells), matched)
        self.assertEqual(digest_file(folder / 'summary.json'), comparison['actual_sha256'])
        self.assertEqual(execution['status'], 'mismatch')
        self.assertEqual(execution['returncode'], 0)
        self.assertEqual(execution['slurm_job_id'], '13725')
        self.assertEqual(comparison['target_coverage']['expected_test_images'], images)

    def test_anonymous_aa_release_matches_pinned_archive_and_raw_maps(self):
        folder = ROOT / 'validation/a10-20261009'
        index = json.loads((folder / 'aa-anonymous-evidence.json').read_text())
        releases = json.loads((ROOT / 'aa-serving-releases.json').read_text())['releases']
        for item in index['checks']:
            for entry in item['files']:
                self.assertEqual(digest_file(folder / entry['file']), entry['sha256'])
            report = json.loads((folder / item['files'][0]['file']).read_text())
            pinned = releases[item['release']]
            self.assertEqual(report['status'], 'matched')
            self.assertEqual(report['hf_revision'], pinned['revision'])
            self.assertEqual(report['sha256'], pinned['sha256'])
            self.assertEqual(report['bytes'], pinned['bytes'])
            self.assertTrue(report['authentication'].startswith('none'))
            self.assertEqual(report['unpack']['files'], pinned['files'])
            self.assertEqual(report['unpack']['recipe'], pinned['recipe'])
            parity = json.loads((folder / item['files'][1]['file']).read_text())
            self.assertEqual(report['parity'], parity)
            self.assertEqual(parity['status'], 'matched')
            self.assertEqual(parity['engine']['artifact_sha256'], pinned['export_sha256'])
            self.assertEqual({r['category'] for r in parity['rows']}, {'01', '02', '03'})
            for row in parity['rows']:
                self.assertEqual(row['host_max_abs_error'], 0)
                self.assertEqual(row['cted_max_abs_error'], 0)

    def test_fresh_aa_cpu_environment_and_both_raw_map_checks(self):
        folder = ROOT / 'validation/a10-20261009'
        index = json.loads((folder / 'aa-clean-cpu-evidence.json').read_text())
        for entry in index['files']:
            self.assertEqual(digest_file(folder / entry['file']), entry['sha256'])
        environment = json.loads((folder / 'aa-public-cpu-environment.json').read_text())
        self.assertIsNone(environment['cuda_build'])
        self.assertFalse(environment['cuda_available'])
        self.assertEqual(environment['torch'], '2.9.1+cpu')
        self.assertEqual(environment['freeze_sha256'], digest_file(folder / 'aa-public-cpu-env.freeze.txt'))
        for name, artifact in [('main','58954532ae2ab2097e33b2a46f80011f0412339b098dc95d336bd1d09627fb55'),
                               ('weak','e1576d5236ba7801ef7ef9f6f6d3064e60170b330631106610025399432790d5')]:
            report = json.loads((folder / ('aa-public-cpu-' + name + '-parity.json')).read_text())
            self.assertEqual(report['status'], 'matched')
            self.assertEqual(report['engine']['artifact_sha256'], artifact)
            self.assertEqual(len(report['rows']), 3)
            for row in report['rows']:
                self.assertEqual(row['host_max_abs_error'], 0)
                self.assertEqual(row['cted_max_abs_error'], 0)

    def test_gateway_evidence_covers_each_pinned_release_and_fixture(self):
        folder = ROOT / 'validation/a10-20261009'
        index = json.loads((folder / 'gateway-evidence.json').read_text())
        self.assertEqual(digest_file(folder / index['file']), index['sha256'])
        report = json.loads((folder / index['file']).read_text())
        registry_path = folder / index['registry_file']
        self.assertEqual(digest_file(registry_path), report['registry_sha256'])
        registry = json.loads(registry_path.read_text(encoding='utf-8'))
        releases = {m['id']: m['artifact_sha256'] for m in registry['models']}
        self.assertEqual(report['status'], 'matched')
        self.assertEqual(report['cases'], 9)
        self.assertEqual(len(report['rows']), 9)
        self.assertEqual({(r['model'], r['fixture_category']) for r in report['rows']},
                         {(model, category) for model in releases for category in ['01', '02', '03']})
        for row in report['rows']:
            self.assertEqual(row['artifact_sha256'], releases[row['model']])
            for key in ['host_max_abs_error', 'cted_max_abs_error', 'raw_image_score_abs_error']:
                self.assertEqual(row[key], 0)
        self.assertEqual(report['unknown_model_status'], 404)
        self.assertEqual(report['missing_aa_category_status'], 422)

    def test_aa_serving_evidence_retains_cross_host_difference_and_local_parity(self):
        folder = ROOT / 'validation/a10-20261009'
        index = json.loads((folder / 'aa-serving-evidence.json').read_text())
        for entry in index['files']:
            self.assertEqual(digest_file(folder / entry['file']), entry['sha256'])
        for name in ['aa-main-engine-parity.json', 'aa-main-relocation-parity.json',
                     'aa-pilab-original-math.json', 'aa-pilab-container-local-parity.json',
                     'aa-pilab-main-original-math.json', 'aa-pilab-main-container-local-parity.json']:
            report = json.loads((folder / name).read_text())
            self.assertEqual(report['status'], 'matched')
            self.assertEqual({r['category'] for r in report['rows']}, {'01', '02', '03'})
            for row in report['rows']:
                self.assertEqual(row['host_max_abs_error'], 0)
                self.assertEqual(row['cted_max_abs_error'], 0)
                if 'raw_image_score_abs_error' in row:
                    self.assertEqual(row['raw_image_score_abs_error'], 0)
        difference = json.loads((folder / 'aa-pilab-a10-difference.json').read_text())
        self.assertEqual(difference['status'], 'mismatch')
        self.assertGreater(max(r['host_max_abs_error'] for r in difference['rows']), 0)
        runtime = json.loads((folder / 'aa-pilab-container-runtime.json').read_text())
        self.assertTrue(runtime['read_only'])
        self.assertEqual(runtime['ports']['8000/tcp'][0]['HostIp'], '127.0.0.1')

    def test_verified_release_matches_archived_references_and_full_target_scope(self):
        folder = ROOT / 'validation/a10-20261009'
        report = json.loads((folder / 'verified-results.json').read_text())
        self.assertEqual(len(report['results']), 18)
        for row in report['results']:
            with self.subTest(recipe=row['recipe']):
                self.assertEqual(row['status'], 'matched')
                for item in row['evidence']:
                    self.assertEqual(digest_file(folder / item['file']), item['sha256'])
                run = (folder / row['evidence'][0]['file']).parent
                execution = json.loads((run / 'execution.json').read_text())
                self.assertEqual(digest_file(run / 'summary.json'), execution['comparison']['actual_sha256'])
                reference, expected = reference_summary(ROOT, row['recipe'])
                actual = json.loads((run / 'summary.json').read_text())
                cells = compare(extract(actual, reference['host']), extract(expected, reference['host']))
                self.assertEqual(cells, execution['comparison']['cells'])
                self.assertTrue(all(c['matches_printed_precision'] for c in cells))
                self.assertEqual(len(cells), row['metrics_checked'])
                if row['target_coverage']['dataset'] == 'mvtec':
                    self.assertEqual(row['target_coverage']['classes_checked'], 15)
                    self.assertEqual(row['target_coverage']['expected_test_images'], 1725)
                    self.assertTrue(row['target_coverage']['reported_image_counts_checked'])
                if row['target_coverage']['dataset'] == 'visa':
                    self.assertEqual(row['target_coverage']['classes_checked'], 12)
                    self.assertEqual(row['target_coverage']['expected_test_images'], 2162)
                    self.assertTrue(row['target_coverage']['reported_image_counts_checked'])
                self.assertFalse(row['target_coverage']['per_image_execution_verified'])

    def test_rawclip_relocated_image_outputs_bind_exact_engine_and_terminal_state(self):
        folder = ROOT / 'validation/a10-20261009/rawclip-relocated-v1'
        index = json.loads((folder / 'index.json').read_text())
        self.assertEqual(index['image_cases'], 9)
        for entry in index['files']:
            self.assertEqual(digest_file(folder / entry['file']), entry['sha256'])
        for model in ['openai', 'l336', 'h14']:
            report = json.loads((folder / model / 'parity.json').read_text())
            export = json.loads((folder / model / 'export/manifest.json').read_text())
            execution = json.loads((folder / model / 'export/execution.json').read_text())
            self.assertEqual(report['status'], 'matched')
            self.assertEqual(report['engine_sha256'], digest_file(ROOT.parent / 'ted/inference/rawclip_engine.py'))
            self.assertEqual(report['artifact_sha256'], digest_file(folder / model / 'export/manifest.json'))
            self.assertTrue(report['original_workspace_reads_denied'])
            self.assertTrue(report['network_denied'])
            self.assertFalse(report['gpu_used'])
            self.assertEqual({r['category'] for r in report['cases']}, {'01', '02', '03'})
            for row in report['cases']:
                self.assertEqual(set(row['map_max_abs_error']), {'host_map', 'tted_map', 'cted_map'})
                self.assertTrue(all(error == 0 for error in row['map_max_abs_error'].values()))
                self.assertTrue(all(error == 0 for error in row['score_abs_error'].values()))
            self.assertEqual(execution['status'], 'matched')
            for entry in export['captured_state']:
                self.assertIn({k: entry[k] for k in ['function', 'file', 'sha256']}, execution['captured_files'])

    def test_rawclip_pilab_docker_preserves_all_three_readouts_on_canonical_images(self):
        folder = ROOT / 'validation/a10-20261009/rawclip-pilab-v1'
        index = json.loads((folder / 'index.json').read_text())
        for entry in index['files']:
            self.assertEqual(digest_file(folder / entry['file']), entry['sha256'])
        for mode in ['relocated', 'http']:
            report = json.loads((folder / f'rawclip-openai-pilab-{mode}-parity-20261009-v1.json').read_text())
            self.assertEqual(report['status'], 'matched')
            self.assertTrue(report['original_workspace_reads_denied'])
            self.assertFalse(report['gpu_used'])
            self.assertEqual(len(report['cases']), 3)
            for row in report['cases']:
                self.assertEqual(set(row['map_max_abs_error']), {'host_map', 'tted_map', 'cted_map'})
                self.assertTrue(all(e == 0 for e in row['map_max_abs_error'].values()))
                self.assertTrue(all(e == 0 for e in row['score_abs_error'].values()))
            if mode == 'http':
                self.assertEqual(report['allowed_http_worker'], 'http://127.0.0.1:18090')
        container = json.loads((folder / 'container.json').read_text())
        self.assertEqual(container['health'], 'healthy')
        self.assertTrue(container['read_only'])
        self.assertEqual(container['ports']['8000/tcp'][0]['HostIp'], '127.0.0.1')
        self.assertEqual(container['environment']['TED_ENGINE_FAMILY'], 'rawclip')
        self.assertEqual(container['environment']['CUDA_VISIBLE_DEVICES'], '')

    def test_a10_published_bytes_and_claims_match_original_execution_records(self):
        folder = ROOT / 'validation/a10-20261009'
        report = json.loads((folder / 'report.json').read_text())
        self.assertEqual(len(report['results']), 23)
        raw_h14 = next(row for row in report['results'] if row['recipe'] == 'rawclip_vith14_mvtec2btad')
        self.assertEqual(raw_h14['status'], 'matched')
        self.assertEqual(raw_h14['metrics_matched_2dp'], 12)
        l336_bayes = next(row for row in report['results'] if row['recipe'] == 'bayespfl-vitl336-mvtec2btad-seed0')
        self.assertEqual(l336_bayes['status'], 'mismatch')
        self.assertEqual(l336_bayes['metrics_matched_2dp'], 0)
        for recipe in ['bayespfl-vith14-mvtec2btad-seed0', 'bayespfl-vitl14openai-mvtec2btad-seed0']:
            rows = [row for row in report['results'] if row['recipe'] == recipe]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['status'], 'mismatch')
            self.assertEqual(rows[0]['metrics_matched_2dp'], 7)
        bayes = [row for row in report['results'] if row['recipe'] == 'bayespfl-vitb_plus-mvtec2btad-seed0']
        self.assertEqual(len(bayes), 1)
        self.assertEqual(bayes[0]['status'], 'mismatch')
        self.assertEqual(bayes[0]['metrics_matched_2dp'], 6)
        adaptclip = [row for row in report['results'] if row['recipe'] == 'adaptclip-vitl14openai-mvtec2btad-seed0']
        self.assertEqual(len(adaptclip), 1)
        self.assertEqual(adaptclip[0]['metrics_matched_2dp'], 8)
        l336 = [row for row in report['results'] if row['recipe'] == 'adaptclip-vitl336-mvtec2btad-seed0']
        self.assertEqual(len(l336), 1)
        self.assertEqual(l336[0]['metrics_matched_2dp'], 8)
        import hashlib
        import zipfile
        with zipfile.ZipFile(folder / 'adaptclip-l336-btad-v5/preparation-and-results.zip') as archive:
            execution = json.loads(archive.read('execution.json'))
            self.assertEqual(hashlib.sha256(archive.read('run.json')).hexdigest(), execution['plan_sha256'])
            self.assertEqual(archive.read('results/summary.json'), (folder / 'adaptclip-l336-btad-v5/summary.json').read_bytes())
            self.assertTrue(archive.read('execution.log'))
        for row in report['results']:
            with self.subTest(recipe=row['recipe']):
                for item in row['evidence']:
                    self.assertEqual(digest_file(folder / item['file']), item['sha256'])
                run = (folder / row['evidence'][0]['file']).parent
                execution = json.loads((run / 'execution.json').read_text())
                comparison = json.loads((run / 'comparison.json').read_text())
                self.assertEqual(digest_file(run / 'summary.json'), execution['comparison']['actual_sha256'])
                self.assertEqual(comparison, execution['comparison'])
                reference, expected = reference_summary(ROOT, row['recipe'])
                actual = json.loads((run / 'summary.json').read_text())
                cells = compare(extract(actual, reference['host']), extract(expected, reference['host']))
                self.assertEqual(cells, comparison['cells'])
                matched = sum(c['matches_printed_precision'] for c in cells)
                self.assertEqual(matched, row['metrics_matched_2dp'])
                self.assertEqual(len(cells), row['metrics_checked'])
                self.assertEqual(execution['status'], 'matched' if matched == len(cells) else 'mismatch')
