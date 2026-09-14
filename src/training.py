"""Sequential full-data LoRA training with resumable optimizer/RNG state."""
import json
import math
import os
import random
import time
from pathlib import Path

try:
    from .common import digest, read_jsonl, seed_everything, write_json
    from .modeling import (autocast, causal_nll_sum, collate, encode_document,
                           encode_sanitizer, evaluate_loss, load_model, load_tokenizer)
except ImportError:
    from common import digest, read_jsonl, seed_everything, write_json
    from modeling import (autocast, causal_nll_sum, collate, encode_document,
                          encode_sanitizer, evaluate_loss, load_model, load_tokenizer)


def training_arguments(parser, default_epochs):
    parser.add_argument("--epochs", type=int, default=default_epochs)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--save_every", type=int, default=25)
    parser.add_argument("--projection_chunk_tokens", type=int, default=128)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--probe", action="store_true", help="Two optimizer steps on synthetic stress-test text; not assignment results.")
    parser.add_argument("--probe_tokens", type=int, default=1024, help="Synthetic stress-test sequence length; ignored during real training.")
    parser.add_argument("--optimizer_device", choices=("cuda", "cpu"), default="cuda",
                        help="Keep the same setting across all six runs. CPU uses FP32 master copies and AdamW states.")


def train(args, *, sanitizer):
    try:
        from .progress import BudgetStop, Journal, stop_due
    except ImportError:
        from progress import BudgetStop, Journal, stop_due
    import numpy as np
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from transformers import get_linear_schedule_with_warmup
    if not 1 <= args.epochs <= 10:
        raise ValueError("Epochs must be between 1 and 10.")
    if min(args.batch_size, args.gradient_accumulation_steps) < 1:
        raise ValueError("Batch size and accumulation must be positive.")
    if args.projection_chunk_tokens < 1:
        raise ValueError("Projection chunk size must be positive.")
    seed_everything(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    corpus = args.train_file if sanitizer else args.corpus
    validation = args.val_file if sanitizer else args.heldout
    quant = getattr(args, "quant", "none")
    specification = {**vars(args), "task": "sanitizer" if sanitizer else "victim",
                     "quant": quant, "corpus_sha256": digest(corpus),
                     "validation_sha256": digest(validation), "row_order": "supplied; never shuffled",
                     "loss": "completion_only" if sanitizer else "all_content_and_eos",
                     "optimizer": "torch.optim.AdamW; betas=(0.9,0.999); eps=1e-8; weight_decay=0",
                     "precision": "fp16_autocast; fp32_adapters; GradScaler",
                     "amp_overflow_policy": "replay identical accumulation group and RNG at half scale; at most 8 attempts; never skip rows",
                     "output_projection_chunk_tokens": args.projection_chunk_tokens}
    state_file = output / "training_state.pt"
    experiment_file = output / "experiment.json"
    if experiment_file.exists():
        previous = json.loads(experiment_file.read_text())
        ignored = {"resume", "save_every", "validation_batch_size"}
        if {k: v for k, v in previous.items() if k not in ignored} != {
                k: v for k, v in specification.items() if k not in ignored}:
            raise ValueError("Existing run has different settings; use a new output directory.")
        if (output / "training_metrics.json").exists() and not args.probe:
            if args.resume:
                return json.loads((output / "training_metrics.json").read_text())
            raise FileExistsError("Completed run exists; pass --resume to reuse it.")
        if not args.resume and state_file.exists():
            raise FileExistsError("Partial run exists; pass --resume or use a new output directory.")
    write_json(experiment_file, specification)
    tokenizer = load_tokenizer(args.model)
    encoder = encode_sanitizer if sanitizer else encode_document
    if args.probe:
        # Build synthetic inputs for the two-step memory probe.
        sample = {"source_text": "Please contact alice@example.org. " * 10,
                  "target_text": "Please contact [EMAIL]. " * 10,
                  "text": "A diagnostic document with no assignment data. " * 20}
        encoded = encoder(tokenizer, sample)
        tokens = encoded["input_ids"]
        probe_ids = (tokens * math.ceil(args.probe_tokens / len(tokens)))[:args.probe_tokens]
        prompt_length = args.probe_tokens // 2 if sanitizer else 1
        stress = {"input_ids": probe_ids, "labels": [-100] * prompt_length + probe_ids[prompt_length:]}
        # Two updates include AdamW states during the second backward pass.
        examples = [stress] * (2 * args.batch_size * args.gradient_accumulation_steps)
        val_examples = [stress]
    else:
        rows, val_rows = read_jsonl(corpus), read_jsonl(validation)
        if sanitizer and (len(rows) != 20000 or len(val_rows) != 2000):
            raise ValueError("Part A requires exactly 20,000 train and 2,000 validation rows.")
        if not sanitizer and (len(rows) != 8007 or len(val_rows) != 500):
            raise ValueError("Part C requires exactly 8,007 training and 500 held-out rows.")
        examples = [encoder(tokenizer, row) for row in rows]
        val_examples = [encoder(tokenizer, row) for row in val_rows]
    maximum = max(len(row["input_ids"]) for row in examples + val_examples)
    if maximum > args.max_length:
        raise ValueError(f"Longest example is {maximum} tokens; --max_length={args.max_length}. Refusing truncation.")
    write_json(output / "tokenization.json", {"max_tokens": maximum, "train_rows": len(examples),
               "validation_rows": len(val_examples), "supervised_tokens_per_epoch":
               sum(sum(x != -100 for x in row["labels"][1:]) for row in examples)})
    model = load_model(args.model, quant=quant, training=True, modules=args.target_modules,
                       rank=args.lora_r, alpha=args.lora_alpha, dropout=getattr(args, "lora_dropout", 0.05))
    model.print_trainable_parameters()
    trainable, total_params = model.get_nb_trainable_parameters()
    model_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer_parameters = ([torch.nn.Parameter(p.detach().cpu().float().clone()) for p in model_parameters]
                            if args.optimizer_device == "cpu" else model_parameters)
    optimizer = torch.optim.AdamW(optimizer_parameters,
                                 lr=args.learning_rate, weight_decay=0.0, foreach=False)
    effective = args.batch_size * args.gradient_accumulation_steps
    steps_per_epoch = math.ceil(len(examples) / effective)
    epochs = 1 if args.probe else args.epochs
    total_steps = epochs * steps_per_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, math.ceil(total_steps * args.warmup_ratio), total_steps)
    scaler = torch.amp.GradScaler("cuda")
    start_step, elapsed, input_tokens, trained_tokens, loss_total, old_peak = 0, 0.0, 0, 0, 0.0, 0
    resumed_last_loss = None
    amp_replays = 0
    if args.resume and state_file.exists():
        state = torch.load(state_file, map_location="cpu", weights_only=False)
        set_peft_model_state_dict(model, state["adapter"])
        if args.optimizer_device == "cpu":
            with torch.no_grad():
                for master, parameter in zip(optimizer_parameters, model_parameters):
                    master.copy_(parameter.detach().cpu())
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_step, elapsed = state["next_step"], state["elapsed"]
        progress_file = output / "progress.json"
        if progress_file.is_file():
            progress = json.loads(progress_file.read_text())
            if progress.get("next_step") == start_step and progress.get("total_steps") == total_steps:
                # The checkpoint payload is assembled before torch.save. Its
                # matching atomic progress record includes that save's I/O time.
                elapsed = max(elapsed, float(progress["elapsed_s"]))
        input_tokens, trained_tokens = state["input_tokens"], state["trained_tokens"]
        loss_total, old_peak = state["loss_total"], state["peak_bytes"]
        resumed_last_loss = state.get("last_loss")
        amp_replays = state.get("amp_replays", 0)
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        del state
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model.train()

    def save_state(next_step):
        torch.cuda.synchronize()
        state = {"adapter": {k: v.detach().cpu() for k, v in get_peft_model_state_dict(model).items()},
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "scaler": scaler.state_dict(), "next_step": next_step,
                 "elapsed": elapsed + time.perf_counter() - started,
                 "input_tokens": input_tokens, "trained_tokens": trained_tokens, "loss_total": loss_total,
                 "last_loss": last_loss,
                 "amp_replays": amp_replays,
                 "peak_bytes": max(old_peak, torch.cuda.max_memory_allocated()),
                 "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                 "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}
        temporary = state_file.with_suffix(".tmp")
        torch.save(state, temporary)
        os.replace(temporary, state_file)

    last_loss = resumed_last_loss
    for step in range(start_step, total_steps):
        if stop_due():
            save_state(step)
            write_json(output / "progress.json", {"status": "paused", "next_step": step,
                       "total_steps": total_steps, "elapsed_s": elapsed + time.perf_counter() - started})
            raise BudgetStop("Training time allowance reached; optimizer and RNG checkpoint saved.")
        offset = (step % steps_per_epoch) * effective
        group = examples[offset:offset + effective]
        group_tokens = sum(sum(x != -100 for x in row["labels"][1:]) for row in group)
        # Dynamic loss scaling may overflow early in FP16 training. Replay the
        # same accumulation group and dropout RNG state at a lower scale.
        # Advance the schedule only after a successful optimizer update.
        group_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
        for attempt in range(8):
            if attempt:
                random.setstate(group_rng[0])
                np.random.set_state(group_rng[1])
                torch.set_rng_state(group_rng[2])
                torch.cuda.set_rng_state_all(group_rng[3])
            optimizer.zero_grad(set_to_none=True)
            model.zero_grad(set_to_none=True)
            step_loss, step_input_tokens, step_trained_tokens = 0.0, 0, 0
            for first in range(0, len(group), args.batch_size):
                micro = group[first:first + args.batch_size]
                batch = collate(micro, tokenizer, model.device)
                with autocast(model):
                    loss, count = causal_nll_sum(model, batch, chunk_size=args.projection_chunk_tokens)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite forward loss at optimizer step {step}.")
                scaler.scale(loss / group_tokens).backward()
                step_loss += float(loss.detach())
                step_input_tokens += sum(len(row["input_ids"]) for row in micro)
                step_trained_tokens += count
                del loss, batch
            if args.optimizer_device == "cpu":
                for master, parameter in zip(optimizer_parameters, model_parameters):
                    master.grad = parameter.grad.detach().cpu() if parameter.grad is not None else None
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(optimizer_parameters, 1.0)
            if torch.isfinite(grad_norm):
                break
            if scaler.get_scale() <= 1 or attempt == 7:
                raise FloatingPointError("Persistent non-finite gradients; no row or optimizer update skipped.")
            scaler.update(new_scale=scaler.get_scale() / 2)
            amp_replays += 1
            print(json.dumps({"step": step + 1, "amp_replay_same_group": attempt + 1,
                              "new_scale": scaler.get_scale()}), flush=True)
        input_tokens += step_input_tokens
        trained_tokens += step_trained_tokens
        scaler.step(optimizer)
        scaler.update()
        if args.optimizer_device == "cpu":
            with torch.no_grad():
                for parameter, master in zip(model_parameters, optimizer_parameters):
                    parameter.copy_(master)
        scheduler.step()
        loss_total += step_loss
        last_loss = step_loss / group_tokens
        if step == start_step or (step + 1) % 10 == 0:
            print(json.dumps({"step": step + 1, "total_steps": total_steps, "loss": last_loss,
                              "elapsed_s": round(elapsed + time.perf_counter() - started, 2)}), flush=True)
        if (step + 1) % args.save_every == 0 or step + 1 == total_steps:
            save_state(step + 1)
            write_json(output / "progress.json", {"status": "training", "next_step": step + 1,
                       "total_steps": total_steps, "elapsed_s": elapsed + time.perf_counter() - started,
                       "remaining_train_s_linear_estimate": (total_steps - step - 1) *
                       (elapsed + time.perf_counter() - started) / (step + 1),
                       "estimate_note": "Productive training only; excludes remaining validation and job overhead."})
    torch.cuda.synchronize()
    wallclock = elapsed + time.perf_counter() - started
    peak = max(old_peak, torch.cuda.max_memory_allocated())
    statistics = {"trainable_params": trainable, "total_params": total_params,
                  "trainable_pct": 100 * trainable / total_params, "peak_gpu_mem_gb": peak / 1024 ** 3,
                  "train_wallclock_s": wallclock, "tokens_per_s": input_tokens / wallclock,
                  "throughput_definition": "non-padding input tokens / training wallclock incl checkpoint I/O",
                  "supervised_tokens_per_s": trained_tokens / wallclock,
                  "final_train_loss": last_loss if last_loss is not None else loss_total / trained_tokens,
                  "mean_train_loss": loss_total / trained_tokens, "optimizer_steps": total_steps,
                  "physical_rows_processed": len(examples) * epochs, "probe_only": args.probe,
                  "amp_overflow_group_replays": amp_replays,
                  "gpu": torch.cuda.get_device_name(), "torch_version": torch.__version__,
                  "execution_concurrency_at_finish": int(os.environ.get("A1_EXECUTION_CONCURRENCY", "1")),
                  "physical_gpu_index_at_finish": os.environ.get("A1_PHYSICAL_GPU_INDEX", "unspecified"),
                  "timing_caveat": "Independent runs may share host CPU/I/O; see immutable Kaggle job receipts for segment concurrency.",
                  "optimizer_device": args.optimizer_device,
                  "model_revision": getattr(model.config, "_commit_hash", None)}
    if stop_due():
        raise BudgetStop("All optimizer steps saved; validation must resume in the next time allowance.")
    validation_journal = Journal(output / "validation_likelihood.progress.json", {
        "checkpoint_sha256": digest(state_file), "validation_sha256": digest(validation),
        "batch_size": args.batch_size, "objective": specification["loss"]}, args.resume)
    validation_metrics = evaluate_loss(model, tokenizer, val_examples, args.batch_size, journal=validation_journal)
    statistics["validation"] = validation_metrics
    if not args.probe:
        model.save_pretrained(output, safe_serialization=True)
        tokenizer.save_pretrained(output)
    write_json(output / ("probe_metrics.json" if args.probe else "training_metrics.json"), statistics)
    write_json(output / "progress.json", {"status": "training_complete", "next_step": total_steps,
               "total_steps": total_steps, "elapsed_s": wallclock})
    return statistics
