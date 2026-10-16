"""Sanitize Part C text while preserving row order and document metadata."""
import argparse
from pathlib import Path

try:
    from .common import digest, read_jsonl, seed_everything, write_json, write_jsonl
    from .modeling import Sanitizer, adapter_quant
    from .progress import Journal
    from .hf_adapters import ensure_adapter, is_complete_adapter
except ImportError:
    from common import digest, read_jsonl, seed_everything, write_json, write_jsonl
    from modeling import Sanitizer, adapter_quant
    from progress import Journal
    from hf_adapters import ensure_adapter, is_complete_adapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    seed_everything(args.seed)
    if not is_complete_adapter("best_sanitizer", Path(args.adapter_path)):
        ensure_adapter("best_sanitizer", args.adapter_path)
    rows = read_jsonl(args.corpus)
    if len(rows) != 8007:
        raise ValueError("Expected all 8,007 physical training rows.")
    try:
        from .modeling import MODEL_REVISIONS, SYSTEM_PROMPT
    except ImportError:
        from modeling import MODEL_REVISIONS, SYSTEM_PROMPT
    quant = adapter_quant(args.adapter_path)
    sources = list(dict.fromkeys(row["text"] for row in rows))
    journal = Journal(str(args.output_file) + ".progress.json", {
        "input_sha256": digest(args.corpus), "adapter_sha256": digest(Path(args.adapter_path) / "adapter_model.safetensors"),
        "base_model": args.base_model, "revision": MODEL_REVISIONS.get(args.base_model), "prompt": SYSTEM_PROMPT,
        "seed": args.seed, "batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens,
        "quant": quant}, args.resume)
    cursor = len(journal.records)
    if cursor > len(sources):
        raise ValueError("Too many cached unique corpus generations.")
    cache = {source: record["output"] for source, record in zip(sources, journal.records)}
    journal.check_stop()
    if cursor < len(sources):
        sanitizer = Sanitizer(args.base_model, args.adapter_path, quant, args.batch_size, args.max_new_tokens)
        for batch in sanitizer.batches(sources[cursor:]):
            for text, latency in batch:
                cache[sources[cursor]] = text
                journal.records.append({"output": text, "latency_ms": latency})
                cursor += 1
            journal.save()
            journal.check_stop()
            print(f"Sanitized {cursor}/{len(sources)} unique texts", flush=True)
    output = []
    # All rows remain present. Only identical text is cached; no canary metadata
    # is consulted. Batch configuration is recorded and fixed across resumes.
    for i, row in enumerate(rows):
        source = row["text"]
        output.append({**row, "text": cache[source]})
        if (i + 1) % 100 == 0:
            print(f"Sanitized {i + 1}/{len(rows)} physical rows", flush=True)
    assert all({k: v for k, v in a.items() if k != "text"} ==
               {k: v for k, v in b.items() if k != "text"} for a, b in zip(rows, output))
    write_jsonl(args.output_file, output)
    write_json(str(args.output_file) + ".provenance.json", {
        "input_sha256": digest(args.corpus), "output_sha256": digest(args.output_file),
        "adapter": args.adapter_path, "rows": len(rows), "distinct_texts": len(cache),
        "decoding": {"do_sample": False, "batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens,
                     "quant": adapter_quant(args.adapter_path)}, "canary_special_cases": False})


if __name__ == "__main__":
    main()
