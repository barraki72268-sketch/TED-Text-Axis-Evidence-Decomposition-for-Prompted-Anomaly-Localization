"""Resolve main and separately scoped weak-source execution configurations."""
import hashlib
import json
from pathlib import Path
import zipfile

from .checkpoint_download import digest_file


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def execution_recipe(root: Path, recipe_id: str) -> dict:
    main = read(root / "execution-recipes.json")["recipes"]
    matches = [r for r in main if r["id"] == recipe_id]
    if matches:
        if len(matches) != 1:
            raise ValueError("Duplicate execution recipe")
        return matches[0]
    manifest = read(root / "ablations/weak-source.json")
    matches = [r for r in manifest["recipes"] if r["id"] == recipe_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown or duplicate recipe: {recipe_id}")
    weak = matches[0]
    dependencies = [r for r in main if r["host"] == weak["host"] and
                    r["transfer"] == weak["preset"] and r["seed"] == weak["seed"] and
                    r["backbone"] == "ViT-L/14-336"]
    if len(dependencies) != 1:
        raise ValueError("Weak-source dependency binding is ambiguous")
    dependency = dependencies[0]
    sources = {s["path"]: s for s in read(root / "source-manifest.json")["files"]}
    launcher = manifest["launcher"]
    if sources[launcher["path"]]["sha256"] != launcher["sha256"]:
        raise ValueError("Weak-source launcher source hash mismatch")
    command = weak["archived_launcher_command"]
    if command[1] != "-u" or command[2] != dependency["evaluator"]["path"]:
        raise ValueError("Weak-source evaluator differs from traced launcher")
    argv = list(command[3:])
    if argv.count("--save_dir") != 1:
        raise ValueError("Expected one weak-source result destination")
    argv[argv.index("--save_dir") + 1] = "{save_dir}"
    for flag, value in (("--preset", weak["preset"]), ("--seed", str(weak["seed"]))):
        if argv.count(flag) != 1 or argv[argv.index(flag) + 1] != value:
            raise ValueError("Weak-source configuration metadata/arguments differ")
    if any(arg.startswith("/") for arg in argv):
        raise ValueError("Unbound absolute weak-source argument")
    if any(arg.split("=", 1)[0] in {"--target_limit_per_class", "--target_class_name", "--class_name"} for arg in argv):
        raise ValueError("Weak-source target coverage must remain unrestricted")
    return dict(dependency, id=recipe_id, argv=argv,
                reference_sha256=weak["reference_sha256"],
                path_bindings={"{save_dir}": {"argument": "--save_dir", "kind": "new_output_directory"}},
                dependency_recipe=dependency["id"], bank_kind="weak", scope="weak-source-ablation",
                argument_provenance="reconstructed archived launcher; not proof of the historical process argv")


def reference_summary(root: Path, recipe_id: str):
    recipe = execution_recipe(root, recipe_id)
    if recipe.get("scope") == "weak-source-ablation":
        manifest = read(root / "ablations/weak-source.json")
        reference = next(r for r in manifest["recipes"] if r["id"] == recipe_id)
        archive = root / "ablations" / manifest["archive"]
        if digest_file(archive) != manifest["archive_sha256"]:
            raise ValueError("Weak-source reference archive hash mismatch")
        with zipfile.ZipFile(archive) as bundle:
            payload = bundle.read(reference["reference"])
        reference = dict(reference, transfer=reference["preset"])
    else:
        reference = next(r for r in read(root / "recipes.json") if r["id"] == recipe_id)
        payload = (root / "references" / reference["reference"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != reference["reference_sha256"]:
        raise ValueError("Historical reference hash mismatch")
    return reference, json.loads(payload)
