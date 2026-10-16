"""Evaluate a sanitizer or baseline on an annotated JSONL dataset."""
import argparse
import time
from pathlib import Path

try:
    from .baselines import PresidioBaseline, regex_sanitize
    from .common import read_jsonl, seed_everything, write_json, write_jsonl, digest
    from .metrics import Metrics
    from .hf_adapters import ensure_adapter, is_complete_adapter
except ImportError:
    from baselines import PresidioBaseline, regex_sanitize
    from common import read_jsonl, seed_everything, write_json, write_jsonl, digest
    from metrics import Metrics
    from hf_adapters import ensure_adapter, is_complete_adapter


def evaluate(args):
    try:
        from .progress import Journal
        from .modeling import MODEL_REVISIONS, SYSTEM_PROMPT, adapter_quant
    except ImportError:
        from progress import Journal
        from modeling import MODEL_REVISIONS, SYSTEM_PROMPT, adapter_quant
    started = time.perf_counter()
    rows = read_jsonl(args.test_file)
    metric, predictions = Metrics(), []
    quant = "none"
    if args.system in {"regex", "presidio"}:
        baseline = regex_sanitize if args.system == "regex" else PresidioBaseline()
        for row in rows:
            start = time.perf_counter()
            output, spans = baseline(row["source_text"])
            predictions.append(metric.add(row, output, spans, 1000 * (time.perf_counter() - start)))
    else:
        try:
            from .modeling import Sanitizer, adapter_quant
        except ImportError:
            from modeling import Sanitizer, adapter_quant
        seed_everything(args.seed)
        adapter = args.adapter_path if args.system == "finetuned" else None
        if args.system == "finetuned" and not adapter:
            raise ValueError("--adapter_path is required for the finetuned system.")
        if args.system == "finetuned" and not is_complete_adapter("best_sanitizer", Path(adapter)):
            adapter = ensure_adapter("best_sanitizer", adapter)
        quant = adapter_quant(adapter) if args.quant == "auto" else args.quant
        identity = {"input_sha256": digest(args.test_file), "model": args.base_model,
                    "model_revision": MODEL_REVISIONS.get(args.base_model), "prompt": SYSTEM_PROMPT,
                    "adapter_sha256": digest(Path(adapter) / "adapter_model.safetensors") if adapter else None,
                    "quant": quant, "batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens,
                    "seed": args.seed, "device": args.device}
        journal = Journal(str(args.output_predictions) + ".progress.json", identity, getattr(args, "resume", False))
        if len(journal.records) > len(rows):
            raise ValueError("Too many saved predictions for this dataset.")
        for row, saved in zip(rows, journal.records):
            if saved["id"] != row["id"]:
                raise ValueError("Saved prediction order differs from the input.")
            predictions.append(metric.add(row, saved["output"], latency_ms=saved["latency_ms"]))
        cursor = len(journal.records)
        resumed_rows = cursor
        journal.check_stop()
        batches = ()
        if cursor < len(rows):
            sanitizer = Sanitizer(args.base_model, adapter, quant, args.batch_size, args.max_new_tokens, args.device)
            batches = sanitizer.batches([row["source_text"] for row in rows[cursor:]])
        for batch in batches:
            for output, latency in batch:
                predictions.append(metric.add(rows[cursor], output, latency_ms=latency))
                journal.records.append({"id": rows[cursor]["id"], "output": output, "latency_ms": latency})
                cursor += 1
            journal.save()
            journal.check_stop()
            if cursor % 100 == 0:
                print(f"Evaluated {cursor}/{len(rows)}", flush=True)
    report = {"system": args.system, "adapter": args.adapter_path if args.system == "finetuned" else None,
              "test_set": Path(args.test_file).stem, "test_sha256": digest(args.test_file),
              "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens,
                           "batch_size": args.batch_size, "quant": quant},
              "label_policy": "strict; no OOD aliases; numbered mask counters removed only after underscore",
              **metric.report()}
    report["seed"] = args.seed
    if args.system == "finetuned":
        report["adapter_sha256"] = digest(Path(args.adapter_path) / "adapter_model.safetensors")
    report["wallclock_s"] = time.perf_counter() - started
    if args.system not in {"regex", "presidio"}:
        report["wallclock_s"] = journal.total_seconds
        report["resumed_generation_rows"] = resumed_rows
    report["within_90_minute_limit"] = report["wallclock_s"] <= 90 * 60
    write_jsonl(args.output_predictions, predictions)
    write_json(args.report_file, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter_path")
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--system", choices=("finetuned", "regex", "presidio", "zeroshot"), required=True)
    parser.add_argument("--output_predictions", required=True)
    parser.add_argument("--report_file", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quant", choices=("auto", "none", "4bit"), default="auto")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = evaluate(args)
    print({k: report[k] for k in ("n", "exact", "overlap_iou50", "leak_rate", "over_masking_rate")})


if __name__ == "__main__":
    main()
