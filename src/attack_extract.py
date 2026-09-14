"""Discoverable, untargeted, canary and targeted extraction attacks."""
import argparse
import contextlib
import hashlib
import json
import math
import random
import struct
import time
import zlib
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

try:
    from .common import contains_value, digest, normalize, read_jsonl, seed_everything, unique_documents, write_json
    from .modeling import (autocast, causal_nll_sum, collate, encode_document, evaluate_loss,
                           generate_ids, load_model, load_tokenizer, per_example_nll, cached_continuation_nll, MODEL_REVISIONS)
    from .progress import Journal
    from .pii_scoring import recovery_metrics, finish_pii_scoring
except ImportError:
    from common import contains_value, digest, normalize, read_jsonl, seed_everything, unique_documents, write_json
    from modeling import (autocast, causal_nll_sum, collate, encode_document, evaluate_loss,
                          generate_ids, load_model, load_tokenizer, per_example_nll, cached_continuation_nll, MODEL_REVISIONS)
    from progress import Journal
    from pii_scoring import recovery_metrics, finish_pii_scoring


def membership_scores(text, victim_nll, reference_nll, lowercase_nll):
    if any(not math.isfinite(value) or value < 0 for value in (victim_nll, reference_nll, lowercase_nll)):
        raise FloatingPointError("Invalid likelihood in membership scoring.")
    payload = text.encode("utf-8")
    if not payload:
        raise ValueError("Cannot score an empty generated string.")
    compressed_bits_per_byte = 8 * len(zlib.compress(payload)) / len(payload)
    return {"ppl": math.exp(victim_nll),
            "ratio_ref": math.exp(victim_nll - reference_nll),
            "ratio_lower": math.exp(victim_nll - lowercase_nll),
            "ratio_zlib": victim_nll / compressed_bits_per_byte}


def sampled_exposure(rank, candidates=10000, space=1000000):
    if not 1 <= rank <= candidates:
        raise ValueError("Sampled rank must be in [1, candidates].")
    rank_hat = rank / candidates * space
    return math.log2(space) - math.log2(rank_hat)


def load_pii_index(path, documents):
    """Load gold PII values by document ID.

    Accepted rows contain doc_id and either values (strings or value/label
    objects) or privacy_mask entities. This format is provisional because
    the official index was not included in the course bundle.
    """
    known = {row["doc_id"]: row["text"] for row in documents if not row["is_canary"]}
    index = {}
    for row in read_jsonl(path):
        doc_id = row["doc_id"]
        if doc_id not in known or doc_id in index:
            raise ValueError(f"Unknown/duplicate PII-index document: {doc_id}")
        values = row.get("values", row.get("privacy_mask"))
        if not isinstance(values, list):
            raise ValueError("PII index needs an explicit 'values' or 'privacy_mask' list per document.")
        entries = [entry if isinstance(entry, str) else entry["value"] for entry in values]
        if any(not value or not contains_value(known[doc_id], value) for value in entries):
            raise ValueError(f"PII index value not present in document {doc_id}")
        index[doc_id] = entries
    if set(index) != set(known):
        raise ValueError("PII index must cover all 5,000 distinct non-canary training documents, including empty lists.")
    return index


class WindowIndex:
    """Hashed token windows with collision verification against the source tokens."""
    def __init__(self, tokenizer, documents, width=50):
        self.width, self.tokens, self.index = width, {}, defaultdict(list)
        for row in documents:
            if row["is_canary"]:
                continue
            tokens = tokenizer.encode(row["text"], add_special_tokens=False)
            self.tokens[row["doc_id"]] = tokens
            for start in range(len(tokens) - width + 1):
                self.index[self.key(tokens[start:start + width])].append((row["doc_id"], start))

    @staticmethod
    def key(tokens):
        return hashlib.blake2b(struct.pack(f"<{len(tokens)}I", *tokens), digest_size=16).digest()

    def matches(self, tokens):
        found = set()
        for start in range(len(tokens) - self.width + 1):
            window = tokens[start:start + self.width]
            for doc_id, offset in self.index.get(self.key(window), []):
                if self.tokens[doc_id][offset:offset + self.width] == window:
                    found.add(doc_id)
        return found


class Attacker:
    def __init__(self, args):
        self.args = args
        self.tokenizer = load_tokenizer(args.victim_model)
        self.model = load_model(args.victim_model, adapter=args.adapter_path, device=args.device)
        self.docs = unique_documents(read_jsonl(args.corpus))
        self.bos = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        self.identity = {"model": args.victim_model, "revision": MODEL_REVISIONS[args.victim_model],
                         "adapter_sha256": digest(Path(args.adapter_path) / "adapter_model.safetensors") if args.adapter_path else None,
                         "seed": args.seed, "device": args.device}
        self.operation_seconds = {}

    def journal(self, name, inputs, settings):
        payload = json.dumps(inputs, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        identity = {**self.identity, "inputs_sha256": hashlib.sha256(payload).hexdigest(), **settings}
        # Attack continuations may contain PII; store them outside public outputs.
        folder = Path(self.args.progress_dir) if self.args.progress_dir else Path("private/attack-progress") / Path(self.args.report_file).stem
        return Journal(folder / f"{name}.json", identity, self.args.resume)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def generate(self, prompts, *, sampled=False, length=64, operation):
        import torch
        journal = self.journal(operation, prompts, {"sampled": sampled, "length": length,
                               "batch_size": self.args.batch_size,
                               "sampling": "temperature1_topk40_topp1" if sampled else "greedy"})
        if len(journal.records) > len(prompts) or (len(journal.records) < len(prompts) and len(journal.records) % self.args.batch_size):
            raise ValueError("Generation checkpoint is not at a valid batch boundary.")
        if sampled and journal.records:
            if journal.state is None:
                raise ValueError("Sampled generation checkpoint lacks RNG state.")
            torch.set_rng_state(torch.tensor(journal.state["torch_rng"], dtype=torch.uint8))
            saved_cuda = journal.state["cuda_rng"]
            if len(saved_cuda) != torch.cuda.device_count():
                raise ValueError("CUDA topology changed during sampled-generation resume.")
            if saved_cuda:
                torch.cuda.set_rng_state_all([torch.tensor(state, dtype=torch.uint8) for state in saved_cuda])
        for start in range(len(journal.records), len(prompts), self.args.batch_size):
            journal.check_stop()
            journal.records.extend(generate_ids(self.model, self.tokenizer, prompts[start:start + self.args.batch_size],
                                                sampled=sampled, max_new_tokens=length))
            if sampled:
                journal.state = {"torch_rng": torch.get_rng_state().tolist(),
                                 "cuda_rng": [state.tolist() for state in torch.cuda.get_rng_state_all()]}
            journal.save()
            if len(journal.records) % 100 == 0:
                print(f"{operation}: {len(journal.records)}/{len(prompts)}", flush=True)
        self.operation_seconds[operation] = journal.total_seconds
        return journal.records

    def nll(self, texts, *, reference=False, operation):
        """Mean likelihood of content tokens, conditioned on an explicit BOS; no added EOS."""
        context = self.model.disable_adapter() if reference and self.args.adapter_path else contextlib.nullcontext()
        examples = []
        for text in texts:
            ids = [self.bos] + self.tokenizer.encode(text, add_special_tokens=False)
            examples.append({"input_ids": ids, "labels": [-100] + ids[1:]})
        journal = self.journal(operation, texts, {"reference": reference, "batch_size": self.args.score_batch_size})
        if len(journal.records) > len(texts):
            raise ValueError("Too many cached likelihoods.")
        with context:
            for start in range(len(journal.records), len(examples), self.args.score_batch_size):
                journal.check_stop()
                journal.records.extend(row["mean_nll"] for row in per_example_nll(
                    self.model, self.tokenizer, examples[start:start + self.args.score_batch_size], self.args.score_batch_size))
                journal.save()
        self.operation_seconds[operation] = journal.total_seconds
        return journal.records

    def discoverable(self):
        prefix, suffix = self.args.prefix_tokens, self.args.suffix_tokens
        included, excluded, prompts, gold = [], 0, [], []
        for row in self.docs:
            if row["is_canary"]:
                continue
            ids = self.tokenizer.encode(row["text"], add_special_tokens=False)
            if len(ids) < prefix + suffix:
                excluded += 1
                continue
            included.append(row)
            prompts.append([self.bos] + ids[:prefix])
            gold.append(ids[prefix:prefix + suffix])
        generated = self.generate(prompts, length=suffix, operation="discoverable-generation")
        by_reps, similarities = defaultdict(list), []
        for row, prediction, expected in zip(included, generated, gold):
            by_reps[str(row["n_reps"])].append(prediction == expected)
            similarities.append(SequenceMatcher(None, prediction, expected, autojunk=False).ratio())
        count = len(included)
        return {"n_documents": count, "n_excluded_short": excluded,
                "exact_match_rate": sum(sum(v) for v in by_reps.values()) / count if count else 0.0,
                "mean_edit_sim": sum(similarities) / count if count else 0.0,
                "by_n_reps": {k: {"n": len(v), "exact_match_rate": sum(v) / len(v)} for k, v in by_reps.items()},
                "protocol": {"prefix_tokens": prefix, "suffix_tokens": suffix,
                             "unit": "distinct non-canary document", "similarity": "SequenceMatcher token ratio",
                             "reference_corpus": "original raw text, identical across all three models"}}

    def untargeted(self, pii_index):
        seed_everything(self.args.seed + 200)
        samples = self.generate([[self.bos]] * self.args.num_samples, sampled=True,
                                length=self.args.untargeted_tokens, operation="untargeted-generation")
        texts = list(dict.fromkeys(self.decode(ids) for ids in samples))
        empty = sum(not text for text in texts)
        texts = [text for text in texts if text]
        victim = self.nll(texts, operation="untargeted-victim-nll")
        reference = self.nll(texts, reference=True, operation="untargeted-reference-nll") if self.args.adapter_path else victim
        lowercase = self.nll([text.lower() for text in texts], operation="untargeted-lowercase-nll")
        scores = [membership_scores(text, a, b, c) for text, a, b, c in zip(texts, victim, reference, lowercase)]
        original = WindowIndex(self.tokenizer, self.docs, self.args.membership_window)
        tokenized = [self.tokenizer.encode(text, add_special_tokens=False) for text in texts]
        raw_members = [original.matches(tokens) for tokens in tokenized]
        trained_members = raw_members
        if self.args.training_corpus and Path(self.args.training_corpus).resolve() != Path(self.args.corpus).resolve():
            trained_index = WindowIndex(self.tokenizer, unique_documents(read_jsonl(self.args.training_corpus)),
                                        self.args.membership_window)
            trained_members = [trained_index.matches(tokens) for tokens in tokenized]
        precision, raw_precision = {}, {}
        for score in ("ppl", "ratio_ref", "ratio_lower", "ratio_zlib"):
            ranked = sorted(range(len(texts)), key=lambda i: (scores[i][score], i))[:100]
            precision[score] = sum(bool(trained_members[i]) for i in ranked) / 100 if len(ranked) == 100 else None
            raw_precision[score] = sum(bool(raw_members[i]) for i in ranked) / 100 if len(ranked) == 100 else None
        recovered_docs = set().union(*raw_members) if raw_members else set()
        folder = Path(self.args.progress_dir) if self.args.progress_dir else Path("private/attack-progress") / Path(self.args.report_file).stem
        scoring_file = folder / "untargeted-pii-scoring-input.json"
        write_json(scoring_file, {"identity": self.identity, "corpus_sha256": digest(self.args.corpus), "texts": texts})
        known_members = [row for row in read_jsonl(self.args.targeted_file) if row["split"] == "member"]
        if len(known_members) != 200:
            raise ValueError("Known-value diagnostic requires the unchanged 200 supplied member prompts.")
        known_values = defaultdict(list)
        raw_by_id = {row["doc_id"]: row["text"] for row in self.docs}
        for row in known_members:
            if row["doc_id"] not in raw_by_id or not contains_value(raw_by_id[row["doc_id"]], row["gold_value"]):
                raise ValueError("A supplied member PII value is not present in its raw document.")
            known_values[row["doc_id"]].append(row["gold_value"])
        known_subset = {**recovery_metrics(texts, known_values), "member_prompts": len(known_members),
            "distinct_documents": len(known_values), "gold_source_sha256": digest(self.args.targeted_file),
            "scope": "SECONDARY: only known values in supplied member prompts; not the full-corpus PII recovery metric"}
        return {"n_generated": len(samples), "n_unique": len(texts), "n_unique_empty_excluded": empty,
                "precision_at_100": precision, "precision_at_100_original_raw": raw_precision,
                "distinct_docs_recovered": len(recovered_docs),
                **recovery_metrics(texts, pii_index),
                "pii_scoring_artifact": str(scoring_file), "pii_scoring_artifact_sha256": digest(scoring_file),
                "pii_index_sha256": digest(self.args.pii_index) if pii_index is not None else None,
                "targeted_gold_subset_recovery": known_subset,
                "protocol": {"sampling": "BOS; temperature=1; top_k=40; top_p=1",
                             "max_new_tokens": self.args.untargeted_tokens,
                             "membership": f"at least one exact {self.args.membership_window}-token window in a non-canary document",
                             "ranking": "ascending; unique decoded non-empty outputs; stable ties by generation order",
                             "pii_denominator": "distinct casefolded whitespace-normalized indexed raw-corpus values",
                             "baseline_caveat": "V-base corpus matches do not establish original pretraining membership"}}

    def conditional_secret_losses(self, prefix, candidates):
        import torch
        prefix_ids = [self.bos] + self.tokenizer.encode(prefix, add_special_tokens=False)
        examples = []
        for secret in candidates:
            full = [self.bos] + self.tokenizer.encode(prefix + secret, add_special_tokens=False)
            if full[:len(prefix_ids)] != prefix_ids:
                raise ValueError("Canary boundary retokenizes the prefix; conditional scoring requires explicit reconciliation.")
            examples.append({"input_ids": full, "labels": [-100] * len(prefix_ids) + full[len(prefix_ids):]})
        continuations = [example["input_ids"][len(prefix_ids):] for example in examples]
        if len({len(tokens) for tokens in continuations}) == 1:
            return cached_continuation_nll(self.model, prefix_ids, continuations, self.args.score_batch_size)
        # Individual secret NLLs from one batched backbone forward.
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        losses = []
        with torch.inference_mode(), autocast(self.model):
            for start in range(0, len(examples), self.args.score_batch_size):
                batch = collate(examples[start:start + self.args.score_batch_size], self.tokenizer, self.model.device)
                states = base.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                    use_cache=False).last_hidden_state[:, :-1]
                labels = batch["labels"][:, 1:]
                for row, target in zip(states, labels):
                    valid = target != -100
                    logits = base.get_output_embeddings()(row[valid]).float()
                    losses.append(float(torch.nn.functional.cross_entropy(logits, target[valid], reduction="sum")))
        return losses

    def canary(self):
        metadata = read_jsonl(self.args.canaries)
        journal = self.journal("canary-ranking", metadata, {"score_batch_size": self.args.score_batch_size,
                               "protocol": "10000_candidates_conditional_secret_nll_pessimistic_ties"})
        by_reps, per_canary = defaultdict(list), []
        for i, row in enumerate(metadata):
            if i < len(journal.records):
                saved = journal.records[i]
                if saved["canary_id"] != row["canary_id"]:
                    raise ValueError("Canary checkpoint order mismatch.")
                per_canary.append(saved)
                by_reps[str(saved["n_reps"])].append(saved["exposure"])
                continue
            journal.check_stop()
            secret = str(row["secret"])
            if len(secret) != 6 or not secret.isascii() or not secret.isdigit():
                raise ValueError("Canary secrets must remain six-digit strings, including leading zeroes.")
            template = row["template"].format(project=row["project"], secret="{secret}")
            if template.count("{secret}") != 1:
                raise ValueError("Expected exactly one secret placeholder.")
            prefix, suffix = template.split("{secret}")
            rng = random.Random(self.args.seed + 1000 + i)
            alternatives = set()
            while len(alternatives) < 9999:
                alternative = f"{rng.randrange(1000000):06d}"
                if alternative != secret:
                    alternatives.add(alternative)
            candidates = [secret] + sorted(alternatives)
            losses = self.conditional_secret_losses(prefix, candidates)
            if len(losses) != 10000 or any(not math.isfinite(value) or value < 0 for value in losses):
                raise FloatingPointError("Expected 10,000 finite nonnegative secret losses; no canary rank is recorded.")
            # Count all tied candidates ahead of the secret (pessimistic rank).
            rank = 1 + sum(value <= losses[0] for value in losses[1:])
            exposure = sampled_exposure(rank)
            by_reps[str(row["n_reps"])].append(exposure)
            per_canary.append({"canary_id": row["canary_id"], "n_reps": row["n_reps"],
                               "sampled_rank": rank, "estimated_rank": rank * 100, "exposure": exposure})
            journal.records.append(per_canary[-1])
            journal.save()
            print(f"Canary {i + 1}/{len(metadata)} ranked", flush=True)
        self.operation_seconds["canary-ranking"] = journal.total_seconds
        return {"n_canaries": len(metadata),
                "mean_exposure": sum(x["exposure"] for x in per_canary) / len(metadata),
                "by_n_reps": {k: {"n": len(v), "mean_exposure": sum(v) / len(v)} for k, v in by_reps.items()},
                "per_canary": per_canary,
                "protocol": "conditional secret-token summed NLL given template prefix; suffix/EOS excluded; pessimistic ties",
                "sampled_estimator_ceiling_bits": math.log2(10000),
                "limitation": "Only 9,999 alternatives sampled; exposure saturates at 13.2877 bits and cannot resolve high true exposure."}

    def targeted(self):
        rows = read_jsonl(self.args.targeted_file)
        if len(rows) != 400:
            raise ValueError("Targeted evaluation requires all 400 supplied prompts.")
        prompts = [[self.bos] + self.tokenizer.encode(row["prompt"], add_special_tokens=False) for row in rows]
        greedy = self.generate(prompts, length=self.args.targeted_tokens, operation="targeted-greedy")
        seed_everything(self.args.seed + 400)
        sampled = self.generate([prompt for prompt in prompts for _ in range(10)], sampled=True,
                                length=self.args.targeted_tokens, operation="targeted-sampled")
        grouped = defaultdict(list)
        without_prompt_gold = defaultdict(list)
        traces = []
        for i, (row, completion) in enumerate(zip(rows, greedy)):
            split = "member" if row["split"] == "member" else "control"
            recovered = contains_value(self.decode(completion), row["gold_value"])
            sampled_recovered = any(contains_value(self.decode(tokens), row["gold_value"])
                                    for tokens in sampled[i * 10:(i + 1) * 10])
            grouped[split].append((recovered, sampled_recovered))
            in_prompt = contains_value(row["prompt"], row["gold_value"])
            if not in_prompt:
                without_prompt_gold[split].append((recovered, sampled_recovered))
            traces.append({"id": row["id"], "doc_id": row["doc_id"], "split": split, "label": row["label"],
                           "greedy_recovered": recovered, "sampled_any10_recovered": sampled_recovered,
                           "gold_already_in_prompt": in_prompt})
        if any(len(grouped[split]) != 200 for split in ("member", "control")):
            raise ValueError("Expected 200 members and 200 controls.")
        return {"n_member": 200, "n_control": 200,
                **{split: {"greedy_leak_rate": sum(x[0] for x in values) / len(values),
                           "sampled_any10_leak_rate": sum(x[1] for x in values) / len(values)}
                   for split, values in grouped.items()}, "per_prompt": traces,
                "excluding_gold_in_prompt": {split: {"n": len(without_prompt_gold[split]),
                    "greedy_leak_rate": sum(row[0] for row in without_prompt_gold[split]) / len(without_prompt_gold[split]) if without_prompt_gold[split] else None,
                    "sampled_any10_leak_rate": sum(row[1] for row in without_prompt_gold[split]) / len(without_prompt_gold[split]) if without_prompt_gold[split] else None}
                    for split in ("member", "control")},
                "protocol": {"match": "generated continuation only; casefold + whitespace normalization",
                             "max_new_tokens": self.args.targeted_tokens, "sampled": "10 draws; T=1; top_k=40; top_p=1"},
                "n_gold_already_in_prompt": sum(row["gold_already_in_prompt"] for row in traces)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--victim_model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--ref_model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--training_corpus", help="Actual V-san training text for membership labels; --corpus stays raw for privacy comparisons.")
    parser.add_argument("--heldout", required=True)
    parser.add_argument("--canaries", required=True)
    parser.add_argument("--pii_index", required=True)
    parser.add_argument("--targeted_file", required=True)
    parser.add_argument("--attack", choices=("discoverable", "untargeted", "canary", "targeted", "all"), default="all")
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_file", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress_dir", help="Private checkpoint directory for attack batches, likelihoods and RNG state.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--score_batch_size", type=int, default=64)
    parser.add_argument("--prefix_tokens", type=int, default=50)
    parser.add_argument("--suffix_tokens", type=int, default=50)
    parser.add_argument("--membership_window", type=int, default=50)
    parser.add_argument("--untargeted_tokens", type=int, default=256)
    parser.add_argument("--targeted_tokens", type=int, default=64)
    parser.add_argument("--allow_missing_pii_index", action="store_true", help="Run without the gold index; full-corpus PII metrics remain null.")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.victim_model != "Qwen/Qwen2.5-0.5B" or args.ref_model != args.victim_model:
        parser.error("Only the specified local base model and its assignment adapters are supported.")
    if args.num_samples != 5000 and args.attack in {"all", "untargeted"}:
        parser.error("The assignment requires exactly 5,000 generated samples.")
    pii_index = None
    if args.attack in {"all", "untargeted"}:
        if Path(args.pii_index).is_file():
            pii_index = load_pii_index(args.pii_index, unique_documents(read_jsonl(args.corpus)))
        elif not args.allow_missing_pii_index:
            parser.error("The official PII index is missing. Supply --pii_index or use --allow_missing_pii_index for partial results.")
    requested = ("discoverable", "untargeted", "canary", "targeted") if args.attack == "all" else (args.attack,)
    # PII labels are used only for scoring. Adding an index can therefore reuse
    # the saved generations and likelihoods without changing their cache keys.
    settings = {key: value for key, value in vars(args).items() if key not in {
        "resume", "report_file", "progress_dir", "pii_index", "allow_missing_pii_index"}}
    fingerprint = {"settings": settings, "inputs": {name: digest(getattr(args, name))
                   for name in ("corpus", "heldout", "canaries", "targeted_file")},
                   "training_corpus": digest(args.training_corpus) if args.training_corpus else None,
                   "adapter": digest(Path(args.adapter_path) / "adapter_model.safetensors") if args.adapter_path else None,
                   "model_revision": MODEL_REVISIONS[args.victim_model]}
    progress_dir = Path(args.progress_dir) if args.progress_dir else Path("private/attack-progress") / Path(args.report_file).stem
    run_journal = Journal(progress_dir / "run.json", fingerprint, args.resume)
    report = run_journal.state or {"victim": Path(args.adapter_path).name if args.adapter_path else "V-base",
              "settings": vars(args), "raw_corpus_sha256": digest(args.corpus),
              "status": "partial", "completed_attacks": [], "attack_elapsed_seconds": {}}
    if set(report["completed_attacks"]) == {"discoverable", "untargeted", "canary", "targeted"}:
        if pii_index is not None:
            report = finish_pii_scoring(report, progress_dir / "untargeted-pii-scoring-input.json", args.pii_index, args.corpus)
        else:
            report.update(status="computed_pending_pii_index", gpu_computations_complete=True)
        run_journal.state = report
        run_journal.save()
        write_json(args.report_file, report)
        print("All GPU attack computations reused; final PII scoring is CPU-only; no model loaded.")
        return
    run_journal.check_stop()
    try:
        seed_everything(args.seed)
        attacker = Attacker(args)
        if "heldout_likelihood" not in report:
            heldout = [encode_document(attacker.tokenizer, row) for row in read_jsonl(args.heldout)]
            utility_journal = attacker.journal("heldout-likelihood", digest(args.heldout),
                {"batch_size": args.score_batch_size, "objective": "all_content_and_eos"})
            utility = evaluate_loss(attacker.model, attacker.tokenizer, heldout, args.score_batch_size, journal=utility_journal)
            report.update(heldout_ppl=utility["ppl"], heldout_likelihood=utility)
            run_journal.state = report
            run_journal.save()
        for name in requested:
            if name in report["completed_attacks"]:
                continue
            run_journal.check_stop()
            seed_everything(args.seed)
            attack_started = time.perf_counter()
            try:
                report[name] = attacker.untargeted(pii_index) if name == "untargeted" else getattr(attacker, name)()
            finally:
                report["attack_elapsed_seconds"][name] = report["attack_elapsed_seconds"].get(name, 0) + time.perf_counter() - attack_started
            report[name]["wallclock_s"] = report["attack_elapsed_seconds"][name]
            report["completed_attacks"].append(name)
            run_journal.state = report
            run_journal.save()
            write_json(args.report_file, report)
        report["gpu_computations_complete"] = args.attack == "all"
        report["status"] = ("complete" if report["untargeted"]["pii_recovery_rate"] is not None
                            else "computed_pending_pii_index") if args.attack == "all" else "partial"
    finally:
        report["total_wallclock_s"] = run_journal.total_seconds
        report["within_180_minute_limit"] = report["total_wallclock_s"] <= 180 * 60
        run_journal.state = report
        run_journal.save()
        write_json(args.report_file, report)


if __name__ == "__main__":
    main()
