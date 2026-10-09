"""Package a passing RawImageBind run without fitting or changing source bytes."""
import argparse
import json
from pathlib import Path
import shutil
from .captured_export import export_captured_run
from .checkpoint_download import digest_file
from .recipe_lookup import execution_recipe

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def build_imagebind_serving_bundle(root, workspace, destination):
    root, workspace, destination = Path(root).resolve(), Path(workspace).resolve(), Path(destination).absolute()
    if destination.exists():
        raise FileExistsError('Bundle destination must be new; preserve previous attempts')
    recipe = execution_recipe(root, read(workspace/'run.json')['recipe'])
    if recipe['host'] != 'RawImageBind':
        raise ValueError('Expected RawImageBind recipe')
    manifest = export_captured_run(root, workspace, destination/'export')
    (destination/'source/ADPretrain/models/ImageBind/weights').mkdir(parents=True,exist_ok=True)
    inventory=[]
    def copy(original, relative, expected):
        path=Path(relative)
        if path.is_absolute() or '..' in path.parts or digest_file(original) != expected:
            raise ValueError('Original serving input changed: '+relative)
        target=destination/path
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(original,target)
        if digest_file(target) != expected:
            raise ValueError('Copied serving input changed: '+relative)
        inventory.append(dict(path=relative,bytes=target.stat().st_size,sha256=expected))
    for entry in manifest['prepared_source_files']:
        copy(workspace/'source'/entry['path'],'source/'+entry['path'],entry['sha256'])
    execution=read(workspace/'execution.json')
    copy(workspace/'run.json','run.json',execution['plan_sha256'])
    catalog=read(root/'backbones.json')
    selected=next(b for b in catalog['bindings'] if b['recipe']==recipe['id'])
    backbone=next(b for b in catalog['artifacts'] if b['sha256']==selected['sha256'])
    copy(workspace/'clip-cache'/backbone['filename'],'clip-cache/'+backbone['filename'],backbone['sha256'])
    report=dict(schema_version=1,host='RawImageBind',recipe=recipe['id'],
        export_sha256=digest_file(destination/'export/manifest.json'),
        original_plan_sha256=execution['plan_sha256'],files=inventory,fitting_performed=False,
        status='packaged_requires_relocated_image_parity_and_public_acquisition',
        scope='Source, backbone, captured bank/calibrators only; no dataset images. Original licenses retained.')
    (destination/'serving-bundle.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace',type=Path)
    parser.add_argument('destination',type=Path)
    args=parser.parse_args()
    print(json.dumps(build_imagebind_serving_bundle(Path(__file__).resolve().parent,args.workspace,args.destination),indent=2))

if __name__=='__main__':
    main()
