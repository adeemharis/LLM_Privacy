# Fine-Tune, Sanitize, Extract

Adeem Haris | 2026JCS2024  
ELL8299 / COL8383 / COL873: Trustworthy Large Language Models  
Assignment 1 | Semester I, 2026-27

Repository: [adeemharis/LLM_Privacy](https://github.com/adeemharis/LLM_Privacy)

Adapters - https://drive.google.com/file/d/1UFh0cbNH1jYzbzzZU9ElNL89z2fAaOM-/view?usp=sharing

Adapters at Hugging Face - https://huggingface.co/adeemharis/LLM_Privacy_Adapters
Victim Adapters at Hugging Face - https://huggingface.co/adeemharis/LLM_Privacy_Victim_Adapters

Part A compares the six specified LoRA/QLoRA configurations for a
Qwen2.5-1.5B-Instruct PII sanitizer. Part B evaluates the selected A4 sanitizer
against regex and Presidio baselines on ID and OOD data. Part C compares
Qwen2.5-0.5B base models trained on raw and sanitized text with an unmodified
base model, using four extraction attacks and held-out perplexity.

## Contents

```text
README.md
report.pdf
requirements.txt
run_all.sh
src/
    peft_sweep.py
    sanitize_eval.py
    train_victim.py
    attack_extract.py
    ... supporting experiment modules
results/
    partA_sweep.csv
    partA_fig.png
    partB_metrics.json
    partC_attacks.json
    partC_exposure.png
```

[report.pdf](report.pdf) explains the methods, comparisons and limitations.
Part B's JSON contains the sanitizer, baseline, validation and OOD metrics.
Part C's JSON contains aggregate results, with V-raw at the top level and all
three models under `comparisons`. It omits per-prompt/per-canary records,
trace records and private file paths; the retained measurements are unchanged.

The complete submission ZIP includes the required
`results/partB_predictions.jsonl`. That file contains source sentences and
per-example predictions, so it is not included here. Course datasets,
canary secrets, extracted continuations, checkpoints, adapter weights and
authentication files are also excluded. Submit the complete ZIP separately
and provide the TAs with private adapter access.

## Setup

Use Python 3.10. Training and model inference require a compatible CUDA GPU;
the recorded experiments used T4 GPUs. Initial downloads need network access.

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
./run_all.sh --stage verify --data_dir data
```

Place the unchanged course JSONL files and `manifest.jsonl` in `data/` first,
or supply another directory using `--data_dir`. The manifest check does not
train a model or use a GPU.

## Run the assignment

Use separate output directories to preserve the included results:

```bash
./run_all.sh --data_dir data --allow_missing_pii_index \
  --results_dir results_rerun --adapters_dir adapters_rerun \
  --work_dir private_rerun
```

This trains A1-A6, evaluates the selected sanitizer, sanitizes the raw corpus,
trains both victims, and runs all four attacks on the three models. The six
required result files, including per-example predictions, are written to
`results_rerun/`. These rerun directories are ignored by Git.

Individual stages are `verify`, `A`, `B`, `sanitize`, `victims`, `attacks` and
`figures`; select one with `--stage`. Use the same directory arguments across
dependent stages. `--resume` needs full local training checkpoints, not just
adapter exports. To redraw the two included plots without model inference:

```bash
./run_all.sh --stage figures
```

The four assignment interfaces are available directly:

```bash
.venv/bin/python src/peft_sweep.py --help
.venv/bin/python src/sanitize_eval.py --help
.venv/bin/python src/train_victim.py --help
.venv/bin/python src/attack_extract.py --help
```

Shared settings are in `src/pipeline.py`: seed 42, one sanitizer epoch, four
victim epochs, learning rate 0.0002, batch size 4 and gradient accumulation 4.
Training defaults in `src/training.py` set 3% warmup and a 1,024-token limit.
Span alignment and metrics are in `src/metrics.py`; baselines are in
`src/baselines.py`; aggregation and plots are in `src/collect_results.py`.

## Private artifacts and limitations

Adapter Zip - Adapters zip can be found at - https://drive.google.com/file/d/1UFh0cbNH1jYzbzzZU9ElNL89z2fAaOM-/view?usp=sharing

Extract the separate private adapter archive and its nested ZIPs into the
project root. The weight files belong at
`adapters/<run>/adapter_model.safetensors`, for A1-A6, `best_sanitizer` (A4),
`victim_raw` and `victim_san`. Stage `B` can reuse `best_sanitizer` without
repeating training. Do not commit these weights, even if they fit GitHub's
file-size limit.

The supplied bundle lacks `partC_pii_index.jsonl`. The flag
`--allow_missing_pii_index` leaves unavailable full-corpus PII recovery metrics
as `null`. If the official index is provided, replace that flag with
`--pii_index /path/to/partC_pii_index.jsonl`. Only one qualifying traced example
was found, and it is weak evidence of training-specific leakage. Evaluation
time overruns are detailed in the report.

## Hugging Face adapters (public sanitizer + protected victim adapters)

The downloader uses a public Hugging Face model repository for Part A sanitizer
adapters (`A1`-`A6` and `best_sanitizer`) and a separate victim-adapter repository
for `victim_raw` and `victim_san`. 

`run_all.sh` downloads the public `best_sanitizer` to
`adapters_dir/hf_downloaded/best_sanitizer` before Part B evaluation or Part C
sanitization. This path is separate from Part A training folders, so training does
not overwrite the downloaded adapter. The `attacks` stage first uses complete
local victim adapters and otherwise downloads them into `hf_downloaded/`; private
victim downloads require access to that private HF repo. A complete `all` run
trains the victim adapters locally before running attacks.

The adapter downloader is also called by the standalone Part B evaluator, Part C
sanitizer, and attack CLI if the requested adapter folder is incomplete.

