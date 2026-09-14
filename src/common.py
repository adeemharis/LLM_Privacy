"""File I/O, data checksums and experiment settings."""
import hashlib
import json
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "trustworthy_llms_a1_bundle" / "data"
ALL_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
SWEEP = [
    ("A1", "q_proj,v_proj", 8, 16, "none"),
    ("A2", "q_proj,k_proj,v_proj,o_proj", 8, 16, "none"),
    ("A3", ALL_MODULES, 8, 16, "none"),
    ("A4", ALL_MODULES, 32, 64, "none"),
    ("A5", ALL_MODULES, 32, 16, "none"),
    ("A6", ALL_MODULES, 32, 64, "4bit"),
]


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize(value):
    return " ".join(str(value).casefold().split())


def contains_value(text, value):
    needle = normalize(value)
    return bool(needle) and needle in normalize(text)


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    import numpy as np
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=False)


def verify_bundle(data_dir=DEFAULT_DATA):
    data_dir = Path(data_dir)
    checks = []
    for item in read_jsonl(data_dir / "manifest.jsonl"):
        path = data_dir / item["file"]
        rows = read_jsonl(path)
        if digest(path) != item["sha256"] or len(rows) != item["rows"]:
            raise ValueError(f"Dataset integrity failure: {path}")
        if path.stat().st_size != item["bytes"]:
            raise ValueError(f"Dataset byte count changed: {path}")
        for row in rows:
            if "privacy_mask" in row:
                source = row["source_text"]
                previous = 0
                parts = []
                for span in row["privacy_mask"]:
                    start, end = span["start"], span["end"]
                    if not previous <= start < end <= len(source):
                        raise ValueError(f"Invalid spans in {row['id']}")
                    if source[start:end] != span["value"]:
                        raise ValueError(f"Offset/value mismatch in {row['id']}")
                    parts.extend((source[previous:start], f"[{span['label']}]"))
                    previous = end
                parts.append(source[previous:])
                if "".join(parts) != row["target_text"]:
                    raise ValueError(f"Target reconstruction failed: {row['id']}")
        checks.append({"file": path.name, "rows": len(rows), "sha256": digest(path)})
    return checks


def unique_documents(rows):
    result = {}
    for row in rows:
        previous = result.setdefault(row["doc_id"], row)
        if previous["text"] != row["text"]:
            raise ValueError(f"Inconsistent repeated document: {row['doc_id']}")
    return list(result.values())
