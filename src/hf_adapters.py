"""Publish/download all Assignment 1 PEFT adapters from Hugging Face Hub.

This configuration makes both repositories public to match the updated
adapter-accessibility instruction: sanitizer adapters (A1-A6 and best_sanitizer)
and Part C victim adapters (victim_raw and victim_san).

Note: the original assignment PDF contains a conflicting restriction on public
victim-model distribution. Confirm the updated instruction with course staff.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

SANITIZER_ADAPTERS = ("A1", "A2", "A3", "A4", "A5", "A6", "best_sanitizer")
VICTIM_ADAPTERS = ("victim_raw", "victim_san")
ALL_ADAPTERS = SANITIZER_ADAPTERS + VICTIM_ADAPTERS
WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")
REQUIRED_FILES = ("adapter_config.json",)
OPTIONAL_FILES = ("experiment.json", "selection.json")

# Replace these two defaults with your actual Hugging Face repository IDs before
# pushing the code, or set A1_HF_SANITIZER_REPO / A1_HF_VICTIM_REPO.
DEFAULT_SANITIZER_REPO_ID = os.environ.get(
    "A1_HF_SANITIZER_REPO", "adeemharis/LLM_Privacy_Adapters"
)
DEFAULT_VICTIM_REPO_ID = os.environ.get(
    "A1_HF_VICTIM_REPO", "adeemharis/LLM_Privacy_Victim_Adapters"
)

ZIP_NAMES = {
    "A1": "A1_adapter_PRIVATE.zip",
    "A2": "A2_adapter_PRIVATE.zip",
    "A3": "A3_adapter_PRIVATE.zip",
    "A4": "A4_adapter_PRIVATE.zip",
    "A5": "A5_adapter_PRIVATE.zip",
    "A6": "A6_adapter_PRIVATE.zip",
    "best_sanitizer": "best_sanitizer_adapter_PRIVATE.zip",
    "victim_raw": "victim_raw_adapter_PRIVATE.zip",
    "victim_san": "victim_san_adapter_PRIVATE.zip",
}


def _is_victim(name: str) -> bool:
    return name in VICTIM_ADAPTERS


def _repo_id(name: str, sanitizer_repo_id: str | None, victim_repo_id: str | None) -> str:
    value = (victim_repo_id or DEFAULT_VICTIM_REPO_ID) if _is_victim(name) else (
        sanitizer_repo_id or DEFAULT_SANITIZER_REPO_ID
    )
    if not value or "YOUR_HF_USERNAME" in value or "/" not in value:
        env_name = "A1_HF_VICTIM_REPO" if _is_victim(name) else "A1_HF_SANITIZER_REPO"
        raise ValueError(
            f"Set {env_name} or replace the corresponding DEFAULT_*_REPO_ID "
            f"in src/hf_adapters.py with a real Hugging Face repo ID. Got: {value!r}"
        )
    return value


def _weight_file(path: Path) -> Path | None:
    for name in WEIGHT_FILES:
        candidate = path / name
        if candidate.is_file():
            return candidate
    return None


def is_complete_adapter(name: str, path: str | Path) -> bool:
    """Return whether a local adapter folder is sufficient for this project."""
    folder = Path(path)
    if not folder.is_dir():
        return False
    if not all((folder / filename).is_file() for filename in REQUIRED_FILES):
        return False
    if _weight_file(folder) is None:
        return False
    # The sanitizer's quantization-aware loader reads this metadata.
    if not _is_victim(name) and not (folder / "experiment.json").is_file():
        return False
    return True


def _safe_experiment(path: Path, name: str) -> dict[str, Any]:
    """Keep only loading/evaluation metadata; omit local paths and CLI dumps."""
    base_model = "Qwen/Qwen2.5-0.5B" if _is_victim(name) else "Qwen/Qwen2.5-1.5B-Instruct"
    source: dict[str, Any] = {}
    candidate = path / "experiment.json"
    if candidate.is_file():
        try:
            parsed = json.loads(candidate.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                source = parsed
        except (OSError, json.JSONDecodeError):
            source = {}
    keep = (
        "quant", "target_modules", "lora_r", "lora_alpha", "lora_dropout",
        "learning_rate", "epochs", "seed", "run_id", "method",
    )
    result = {key: source[key] for key in keep if key in source}
    if "quant" not in result:
        result["quant"] = "4bit" if name == "A6" else "none"
    result["base_model"] = base_model
    result["published_adapter"] = name
    return result


def _locate_zip_member(zf: zipfile.ZipFile, basename: str) -> str | None:
    matches = []
    for member in zf.namelist():
        normalized = member.replace("\\", "/")
        if PurePosixPath(normalized).name == basename and not normalized.endswith("/"):
            matches.append(member)
    if not matches:
        return None
    # Prefer the shallowest path if an archive unexpectedly has duplicates.
    return min(matches, key=lambda member: (len(PurePosixPath(member).parts), member))


def _stage_zip(zip_path: Path, adapter_name: str, destination: Path) -> None:
    if not zip_path.is_file():
        raise FileNotFoundError(f"Missing adapter archive: {zip_path}")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        config_member = _locate_zip_member(zf, "adapter_config.json")
        weight_member = _locate_zip_member(zf, "adapter_model.safetensors")
        if not weight_member:
            weight_member = _locate_zip_member(zf, "adapter_model.bin")
        if not config_member or not weight_member:
            raise ValueError(
                f"{zip_path.name} does not contain adapter_config.json and an adapter weight file."
            )
        (destination / "adapter_config.json").write_bytes(zf.read(config_member))
        weight_name = PurePosixPath(weight_member.replace("\\", "/")).name
        (destination / weight_name).write_bytes(zf.read(weight_member))
        exp_member = _locate_zip_member(zf, "experiment.json")
        if exp_member:
            # Read into memory and write a sanitized metadata file below.
            try:
                (destination / "experiment.json").write_text(
                    json.dumps(json.loads(zf.read(exp_member)), indent=2), encoding="utf-8"
                )
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        selection_member = _locate_zip_member(zf, "selection.json")
        if selection_member and adapter_name == "best_sanitizer":
            try:
                selection = json.loads(zf.read(selection_member))
                (destination / "selection.json").write_text(
                    json.dumps(selection, indent=2), encoding="utf-8"
                )
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
    (destination / "experiment.json").write_text(
        json.dumps(_safe_experiment(destination, adapter_name), indent=2) + "\n",
        encoding="utf-8",
    )


def _ensure_repo_visibility(api: HfApi, repo_id: str, *, private: bool) -> None:
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    info = api.repo_info(repo_id=repo_id, repo_type="model")
    if bool(info.private) != private:
        intended = "private" if private else "public"
        current = "private" if info.private else "public"
        raise RuntimeError(
            f"Hugging Face repo {repo_id} already exists as {current}, but this workflow "
            f"requires it to be {intended}. Change visibility deliberately in the Hub UI, "
            "then rerun the publisher. No files were uploaded to this repo."
        )


def publish_adapters(
    source_dir: str | Path,
    sanitizer_repo_id: str,
    victim_repo_id: str,
    *,
    victims_public: bool = True,
) -> None:
    source_dir = Path(source_dir).expanduser().resolve()
    api = HfApi()
    if sanitizer_repo_id == victim_repo_id:
        raise ValueError(
            "Use separate Hugging Face repositories for sanitizer and victim adapters "
            "so victim visibility can be controlled independently."
        )
    # Confirm local archives before creating/updating any repositories.
    missing = [name for name, filename in ZIP_NAMES.items() if not (source_dir / filename).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing these archives under --source-dir: " + ", ".join(ZIP_NAMES[n] for n in missing)
        )

    with tempfile.TemporaryDirectory(prefix="a1_hf_publish_") as tmp:
        root = Path(tmp)
        sanitizer_stage = root / "sanitizers"
        victim_stage = root / "victims"
        sanitizer_stage.mkdir()
        victim_stage.mkdir()
        for name in SANITIZER_ADAPTERS:
            _stage_zip(source_dir / ZIP_NAMES[name], name, sanitizer_stage / name)
        for name in VICTIM_ADAPTERS:
            _stage_zip(source_dir / ZIP_NAMES[name], name, victim_stage / name)

        (sanitizer_stage / "README.md").write_text(
            "# Assignment 1 sanitizer adapters\n\n"
            "Public PEFT adapters for Part A: A1-A6 and best_sanitizer.\n\n"
            "Base model: `Qwen/Qwen2.5-1.5B-Instruct`.\n",
            encoding="utf-8",
        )
        visibility = "public" if victims_public else "private"
        (victim_stage / "README.md").write_text(
            "# Assignment 1 victim adapters\n\n"
            "Contains the Part C `victim_raw` and `victim_san` PEFT adapters.\n\n"
            f"Repository visibility intended by publisher: {visibility}.\n"
            "Do not distribute publicly unless the course staff explicitly confirms that the original assignment restriction has been superseded.\n",
            encoding="utf-8",
        )

        _ensure_repo_visibility(api, sanitizer_repo_id, private=False)
        _ensure_repo_visibility(api, victim_repo_id, private=not victims_public)
        api.upload_folder(
            folder_path=str(sanitizer_stage), repo_id=sanitizer_repo_id,
            repo_type="model", commit_message="Publish Part A sanitizer adapters A1-A6 and best_sanitizer",
        )
        api.upload_folder(
            folder_path=str(victim_stage), repo_id=victim_repo_id,
            repo_type="model", commit_message="Publish Part C victim adapters",
        )

    print(f"Published public sanitizer adapters: https://huggingface.co/{sanitizer_repo_id}")
    print(f"Published {visibility} victim adapters: https://huggingface.co/{victim_repo_id}")


def ensure_adapter(
    name: str,
    destination: str | Path,
    *,
    sanitizer_repo_id: str | None = None,
    victim_repo_id: str | None = None,
    revision: str = "main",
    force_download: bool = False,
) -> str:
    """Download one adapter into destination without touching training folders."""
    if name not in ALL_ADAPTERS:
        raise ValueError(f"Unknown adapter {name!r}; valid names are: {', '.join(ALL_ADAPTERS)}")
    repo_id = _repo_id(name, sanitizer_repo_id, victim_repo_id)
    destination = Path(destination).expanduser().resolve()
    manifest = destination / ".hf_adapter_source.json"
    expected_source = {"repo_id": repo_id, "subfolder": name, "revision": revision}
    if not force_download and is_complete_adapter(name, destination) and manifest.is_file():
        try:
            if json.loads(manifest.read_text(encoding="utf-8")) == expected_source:
                return str(destination)
        except (OSError, json.JSONDecodeError):
            pass

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"hf_{name}_", dir=destination.parent) as temp:
        stage = Path(temp) / "adapter"
        stage.mkdir()
        for filename in REQUIRED_FILES:
            path = hf_hub_download(
                repo_id=repo_id, repo_type="model", filename=f"{name}/{filename}", revision=revision
            )
            shutil.copy2(path, stage / filename)
        downloaded_weight = None
        for filename in WEIGHT_FILES:
            try:
                path = hf_hub_download(
                    repo_id=repo_id, repo_type="model", filename=f"{name}/{filename}", revision=revision
                )
                downloaded_weight = filename
                shutil.copy2(path, stage / filename)
                break
            except EntryNotFoundError:
                continue
        if not downloaded_weight:
            raise FileNotFoundError(
                f"No adapter_model.safetensors or adapter_model.bin found under {repo_id}/{name}."
            )
        for filename in OPTIONAL_FILES:
            try:
                path = hf_hub_download(
                    repo_id=repo_id, repo_type="model", filename=f"{name}/{filename}", revision=revision
                )
                shutil.copy2(path, stage / filename)
            except EntryNotFoundError:
                pass
        if not _is_victim(name) and not (stage / "experiment.json").is_file():
            # The public best-sanitizer metadata is needed to choose quantization.
            metadata = {"quant": "4bit" if name == "A6" else "none", "published_adapter": name,
                        "base_model": "Qwen/Qwen2.5-1.5B-Instruct"}
            (stage / "experiment.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        (stage / ".hf_adapter_source.json").write_text(
            json.dumps(expected_source, indent=2) + "\n", encoding="utf-8"
        )
        destination.mkdir(parents=True, exist_ok=True)
        # Remove an old alternate weight extension before replacing current files.
        old_weight = "adapter_model.bin" if downloaded_weight == "adapter_model.safetensors" else "adapter_model.safetensors"
        try:
            (destination / old_weight).unlink()
        except FileNotFoundError:
            pass
        for item in stage.iterdir():
            os.replace(item, destination / item.name)
    if not is_complete_adapter(name, destination):
        raise RuntimeError(f"Downloaded adapter {name} is incomplete at {destination}.")
    print(f"Ready: {name} -> {destination} (repo={repo_id}, revision={revision})", flush=True)
    return str(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    publish = sub.add_parser("publish", help="Publish the nine adapter folders from the provided ZIP archives.")
    publish.add_argument("--source-dir", required=True, help="Directory containing the nine *_PRIVATE.zip files.")
    publish.add_argument("--sanitizer-repo-id", default=DEFAULT_SANITIZER_REPO_ID)
    publish.add_argument("--victim-repo-id", default=DEFAULT_VICTIM_REPO_ID)
    visibility = publish.add_mutually_exclusive_group()
    visibility.add_argument(
        "--victims-public", dest="victims_public", action="store_true",
        help="Publish victim adapters publicly (the default).",
    )
    visibility.add_argument(
        "--victims-private", dest="victims_public", action="store_false",
        help="Keep the victim repository private if course staff requires that.",
    )
    publish.set_defaults(victims_public=True)

    download = sub.add_parser("download", help="Download one adapter folder from Hugging Face.")
    download.add_argument("--name", choices=ALL_ADAPTERS, required=True)
    download.add_argument("--destination", required=True)
    download.add_argument("--sanitizer-repo-id", default=DEFAULT_SANITIZER_REPO_ID)
    download.add_argument("--victim-repo-id", default=DEFAULT_VICTIM_REPO_ID)
    download.add_argument("--revision", default="main")
    download.add_argument("--force-download", action="store_true")

    args = parser.parse_args()
    if args.command == "publish":
        publish_adapters(
            args.source_dir, args.sanitizer_repo_id, args.victim_repo_id,
            victims_public=args.victims_public,
        )
    else:
        ensure_adapter(
            args.name, args.destination,
            sanitizer_repo_id=args.sanitizer_repo_id,
            victim_repo_id=args.victim_repo_id,
            revision=args.revision,
            force_download=args.force_download,
        )


if __name__ == "__main__":
    main()
