"""Run the sanitizer experiments, victim training and extraction attacks."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

try:
    from .common import ALL_MODULES, ROOT, SWEEP, verify_bundle, write_json
    from .collect_results import collect_part_b, collect_part_c, plot_part_a, plot_part_c
except ImportError:
    from common import ALL_MODULES, ROOT, SWEEP, verify_bundle, write_json
    from collect_results import collect_part_b, collect_part_c, plot_part_a, plot_part_c


SETTINGS = {
    "seed": 42, "part_a_epochs": 1, "victim_epochs": 4,
    "training_batch_size": 4, "gradient_accumulation_steps": 4,
    "projection_chunk_tokens": 128, "save_every_optimizer_steps": 25,
    "generation_batch_size": 8, "attack_batch_size": 32,
    "score_batch_size": 64, "max_new_tokens": 512,
}


def run(script, *arguments):
    command = [sys.executable, str(ROOT / "src" / script), *map(str, arguments)]
    print("RUN:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def select_best(results, adapters):
    with (results / "partA_sweep.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 6 or {row["run_id"] for row in rows} != {row[0] for row in SWEEP}:
        raise ValueError("Selection requires one complete result for each of A1-A6.")
    highest = max(float(row["val_entity_f1"]) for row in rows)
    candidates = [row for row in rows if float(row["val_entity_f1"]) >= highest - 0.01]
    saved = results / "partB_metrics.json"
    validation = json.loads(saved.read_text()).get("validation", {}) if saved.is_file() else {}

    def key(row):
        metrics = adapters / row["run_id"] / "validation_metrics.json"
        report = json.loads(metrics.read_text()) if metrics.is_file() else validation[row["run_id"]]
        return float(row["val_leak_rate"]), report["over_masking_rate"], float(row["peak_gpu_mem_gb"]), row["run_id"]

    best = min(candidates, key=key)
    source, destination = adapters / best["run_id"], adapters / "best_sanitizer"
    destination.mkdir(parents=True, exist_ok=True)
    selection = destination / "selection.json"
    if selection.exists() and json.loads(selection.read_text())["run_id"] != best["run_id"]:
        raise ValueError("Best checkpoint changed; use a fresh adapter directory for the new experiment.")
    for filename in ("adapter_config.json", "adapter_model.safetensors", "experiment.json"):
        shutil.copy2(source / filename, destination / filename)
    write_json(selection, {"run_id": best["run_id"], "validation_best_f1": highest,
        "rule": "within 0.01 of best validation IoU micro-F1; minimize leak, then overmask, then memory",
        "selected_row": best})
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("verify", "A", "B", "sanitize", "victims", "attacks", "figures", "all"), default="all")
    parser.add_argument("--data_dir", type=Path, default=ROOT / "data")
    parser.add_argument("--results_dir", type=Path, default=ROOT / "results")
    parser.add_argument("--adapters_dir", type=Path, default=ROOT / "adapters")
    parser.add_argument("--work_dir", type=Path, default=ROOT / "private")
    parser.add_argument("--pii_index", type=Path)
    parser.add_argument("--allow_missing_pii_index", action="store_true",
                        help="Run without the gold index; full-corpus PII metrics remain null.")
    parser.add_argument("--resume", action="store_true", help="Resume from full local checkpoints, including optimizer and RNG state.")
    args = parser.parse_args()
    data, results, adapters, work = (path.resolve() for path in
                                   (args.data_dir, args.results_dir, args.adapters_dir, args.work_dir))
    stage = args.stage
    pii = args.pii_index.resolve() if args.pii_index else data / "partC_pii_index.jsonl"
    if stage in {"all", "attacks"} and not pii.is_file() and not args.allow_missing_pii_index:
        parser.error("The official PII index is missing. Supply --pii_index or use --allow_missing_pii_index for partial results.")
    if stage in {"all", "A"} and (results / "partA_sweep.csv").exists() and not args.resume:
        parser.error("Part A results already exist. Use a fresh --results_dir, or --resume with full training checkpoints.")
    if stage in {"verify", "all"}:
        checked = verify_bundle(data)
        print(f"Dataset checks passed for {len(checked)} files.")
    if stage == "verify":
        return
    if stage == "figures":
        plot_part_a(results)
        plot_part_c(results)
        return
    resume = ["--resume"] if args.resume else []
    evaluations = work / "evaluations"
    sanitized = work / "partC_corpus_sanitized.jsonl"
    controls = ["--batch_size", SETTINGS["training_batch_size"],
                "--gradient_accumulation_steps", SETTINGS["gradient_accumulation_steps"],
                "--projection_chunk_tokens", SETTINGS["projection_chunk_tokens"],
                "--save_every", SETTINGS["save_every_optimizer_steps"],
                "--learning_rate", "0.0002", "--seed", SETTINGS["seed"]]
    if stage in {"A", "all"}:
        for run_id, modules, rank, alpha, quant in SWEEP:
            run("peft_sweep.py", "--model", "Qwen/Qwen2.5-1.5B-Instruct",
                "--train_file", data / "partA_train.jsonl", "--val_file", data / "partA_validation.jsonl",
                "--target_modules", modules, "--lora_r", rank, "--lora_alpha", alpha,
                "--lora_dropout", "0.05", "--quant", quant, "--epochs", SETTINGS["part_a_epochs"],
                *controls, "--validation_batch_size", SETTINGS["generation_batch_size"],
                "--run_id", run_id, "--output_dir", adapters / run_id,
                "--results_csv", results / "partA_sweep.csv", *resume)
        select_best(results, adapters)
        plot_part_a(results)
    if stage in {"B", "all"}:
        if not (adapters / "best_sanitizer/adapter_model.safetensors").is_file():
            select_best(results, adapters)
        for system in ("finetuned", "regex", "presidio"):
            for suffix, filename in (("", "partB_test.jsonl"), ("_ood", "partB_ood_test.jsonl")):
                primary = system == "finetuned" and not suffix
                predictions = results / "partB_predictions.jsonl" if primary else work / f"partB{suffix}_{system}_predictions.jsonl"
                run("sanitize_eval.py", "--test_file", data / filename, "--system", system,
                    "--adapter_path", adapters / "best_sanitizer", "--output_predictions", predictions,
                    "--report_file", evaluations / f"partB{suffix}_{system}_metrics.json",
                    "--batch_size", SETTINGS["generation_batch_size"], "--max_new_tokens", SETTINGS["max_new_tokens"],
                    "--seed", SETTINGS["seed"], "--device", "cuda" if system == "finetuned" else "cpu", *resume)
        collect_part_b(evaluations, results, adapters, data)
    if stage in {"sanitize", "all"}:
        run("sanitize_corpus.py", "--corpus", data / "partC_corpus_raw.jsonl", "--output_file", sanitized,
            "--adapter_path", adapters / "best_sanitizer", "--batch_size", SETTINGS["generation_batch_size"],
            "--max_new_tokens", SETTINGS["max_new_tokens"], "--seed", SETTINGS["seed"], *resume)
    if stage in {"victims", "all"}:
        if not sanitized.is_file():
            raise FileNotFoundError("Run --stage sanitize before victim training.")
        for name, corpus in (("raw", data / "partC_corpus_raw.jsonl"), ("san", sanitized)):
            run("train_victim.py", "--model", "Qwen/Qwen2.5-0.5B", "--corpus", corpus,
                "--heldout", data / "partC_heldout.jsonl", "--target_modules", ALL_MODULES,
                "--lora_r", 32, "--lora_alpha", 64, "--lora_dropout", "0.05",
                "--epochs", SETTINGS["victim_epochs"], *controls,
                "--output_dir", adapters / f"victim_{name}", *resume)
    if stage in {"attacks", "all"}:
        if not sanitized.is_file():
            raise FileNotFoundError("The sanitized corpus is needed to score V-san training membership.")
        deferred = ["--allow_missing_pii_index"] if not pii.is_file() else []
        for name in ("base", "raw", "san"):
            adapter = [] if name == "base" else ["--adapter_path", adapters / f"victim_{name}"]
            run("attack_extract.py", "--corpus", data / "partC_corpus_raw.jsonl",
                "--training_corpus", sanitized if name == "san" else data / "partC_corpus_raw.jsonl",
                "--heldout", data / "partC_heldout.jsonl", "--canaries", data / "partC_canaries.jsonl",
                "--pii_index", pii, "--targeted_file", data / "partC_targeted.jsonl",
                "--attack", "all", "--num_samples", 5000, "--seed", SETTINGS["seed"],
                "--batch_size", SETTINGS["attack_batch_size"], "--score_batch_size", SETTINGS["score_batch_size"],
                "--progress_dir", work / f"attack-progress/{name}",
                "--report_file", evaluations / f"partC_{name}_attacks.json", *adapter, *deferred, *resume)
        collect_part_c(evaluations, results, data, sanitized)
        plot_part_c(results)


if __name__ == "__main__":
    main()
