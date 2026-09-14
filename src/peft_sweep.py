"""Train and validate one LoRA/QLoRA sanitizer configuration."""
import argparse
import csv
import gc
import json
import os
from pathlib import Path
from types import SimpleNamespace

try:
    from .training import train, training_arguments
except ImportError:
    from training import train, training_arguments

FIELDS = ["run_id", "method", "target_modules", "lora_r", "lora_alpha", "quant",
          "trainable_params", "total_params", "trainable_pct", "peak_gpu_mem_gb",
          "train_wallclock_s", "tokens_per_s", "final_train_loss", "val_loss",
          "val_entity_f1", "val_leak_rate"]


def append_sweep_row(path, row, resume=False):
    """Atomically append exactly one row under a separate POSIX advisory lock."""
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        existing = []
        if path.exists():
            with path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames != FIELDS:
                    raise ValueError("Unexpected sweep CSV schema.")
                existing = list(reader)
        previous = [item for item in existing if item["run_id"] == row["run_id"]]
        if previous:
            if resume and len(previous) == 1 and all(previous[0][key] == str(row[key]) for key in FIELDS):
                return False
            raise ValueError(f"Duplicate or inconsistent run_id {row['run_id']}")
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(existing + [row])
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--val_file", required=True)
    parser.add_argument("--target_modules", required=True)
    parser.add_argument("--lora_r", type=int, required=True)
    parser.add_argument("--lora_alpha", type=int, required=True)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--quant", choices=("none", "4bit"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--results_csv", required=True)
    parser.add_argument("--validation_batch_size", type=int, default=8)
    training_arguments(parser, 1)
    args = parser.parse_args()
    if args.model != "Qwen/Qwen2.5-1.5B-Instruct":
        parser.error("The assignment requires Qwen/Qwen2.5-1.5B-Instruct.")
    result = train(args, sanitizer=True)
    if args.probe:
        print("Synthetic feasibility probe complete; no assignment CSV row was produced.")
        return
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    try:
        from .sanitize_eval import evaluate
        from .common import digest
    except ImportError:
        from sanitize_eval import evaluate
        from common import digest
    output = Path(args.output_dir)
    metrics_file = output / "validation_metrics.json"
    cached = json.loads(metrics_file.read_text()) if args.resume and metrics_file.exists() else None
    if cached and cached.get("test_sha256") == digest(args.val_file) and cached.get("adapter_sha256") == digest(
            output / "adapter_model.safetensors") and cached.get("seed") == args.seed and cached.get(
            "decoding", {}).get("batch_size") == args.validation_batch_size:
        metrics = cached
    else:
        metrics = evaluate(SimpleNamespace(base_model=args.model, adapter_path=str(output),
                           test_file=args.val_file, system="finetuned", batch_size=args.validation_batch_size, max_new_tokens=512,
                           seed=args.seed, quant="auto", device="cuda",
                           resume=args.resume,
                           output_predictions=str(output / "validation_predictions.jsonl"),
                           report_file=str(metrics_file)))
    row = {k: result[k] for k in FIELDS if k in result}
    row.update({"run_id": args.run_id, "method": "QLoRA" if args.quant == "4bit" else "LoRA",
                "target_modules": args.target_modules, "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
                "quant": args.quant, "val_loss": result["validation"]["mean_nll"],
                "val_entity_f1": metrics["overlap_iou50"]["micro_f1"], "val_leak_rate": metrics["leak_rate"]})
    append_sweep_row(args.results_csv, row, args.resume)


if __name__ == "__main__":
    main()
