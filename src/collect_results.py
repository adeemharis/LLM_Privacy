"""Collect the Part B/C comparisons and plot the two required result figures."""
import csv
import json
import math
from pathlib import Path

try:
    from .common import contains_value, read_jsonl, write_json
except ImportError:
    from common import contains_value, read_jsonl, write_json


def read_json(path):
    return json.loads(Path(path).read_text())


def label_coverage(train_labels, metrics):
    groups = {}
    for name, seen in (("seen", True), ("unseen", False)):
        selected = {label: scores for label, scores in metrics["per_class_overlap_iou50"].items()
                    if scores["support"] > 0 and (label in train_labels) == seen}
        support = sum(scores["support"] for scores in selected.values())
        matched = sum(scores["tp"] for scores in selected.values())
        groups[name] = {"labels": sorted(selected), "n_labels": len(selected),
                        "support": support, "tp": matched,
                        "recall": matched / support if support else None}
    return groups


def collect_part_b(evaluations, results, adapters, data):
    """Keep the fine-tuned ID schema at the top level; add comparisons below it."""
    evaluations, results, adapters, data = map(Path, (evaluations, results, adapters, data))
    metrics = read_json(evaluations / "partB_finetuned_metrics.json")
    metrics["baselines"] = {name: read_json(evaluations / f"partB_{name}_metrics.json")
                            for name in ("regex", "presidio")}
    metrics["ood"] = {name: read_json(evaluations / f"partB_ood_{name}_metrics.json")
                       for name in ("finetuned", "regex", "presidio")}
    validation = {f"A{i}": read_json(adapters / f"A{i}/validation_metrics.json")
                  for i in range(1, 7) if (adapters / f"A{i}/validation_metrics.json").is_file()}
    previous = results / "partB_metrics.json"
    if not validation and previous.is_file():
        validation = read_json(previous).get("validation", {})
    if validation:
        metrics["validation"] = validation
    selection = adapters / "best_sanitizer/selection.json"
    if selection.is_file():
        metrics["selection"] = read_json(selection)
    train_labels = {span["label"] for row in read_jsonl(data / "partA_train.jsonl")
                    for span in row["privacy_mask"]}
    metrics["generalization"] = label_coverage(train_labels, metrics["ood"]["finetuned"])
    write_json(results / "partB_metrics.json", metrics)
    return metrics


def redact_traces(data, sanitized, report, base_report=None, raw_report=None):
    by_id = {row["doc_id"]: row for row in read_jsonl(sanitized)}
    raw_text = {row["doc_id"]: row["text"] for row in read_jsonl(data / "partC_corpus_raw.jsonl")}
    targeted = {row["id"]: row for row in read_jsonl(data / "partC_targeted.jsonl")}
    traces = []
    control = {row["id"]: row for row in base_report["targeted"]["per_prompt"]} if base_report else {}
    raw_control = {row["id"]: row for row in raw_report["targeted"]["per_prompt"]} if raw_report else {}
    for attack in report["targeted"]["per_prompt"]:
        if attack["split"] != "member" or not (attack["greedy_recovered"] or attack["sampled_any10_recovered"]):
            continue
        row = targeted[attack["id"]]
        if attack["gold_already_in_prompt"]:
            continue
        if (contains_value(raw_text[row["doc_id"]], row["gold_value"])
                and contains_value(by_id[row["doc_id"]]["text"], row["gold_value"])):
            traces.append({"prompt_id": row["id"], "doc_id": row["doc_id"], "label": row["label"],
                           "value": "[REDACTED]", "value_length": len(row["gold_value"]), "survived_sanitizer": True,
                           "present_in_raw_training_text": True,
                           "greedy_recovered": attack["greedy_recovered"],
                           "sampled_any10_recovered": attack["sampled_any10_recovered"],
                           "prompt_already_contained_value": attack["gold_already_in_prompt"],
                           "base_greedy_recovered": control.get(attack["id"], {}).get("greedy_recovered"),
                           "base_sampled_any10_recovered": control.get(attack["id"], {}).get("sampled_any10_recovered"),
                           "raw_greedy_recovered": raw_control.get(attack["id"], {}).get("greedy_recovered"),
                           "raw_sampled_any10_recovered": raw_control.get(attack["id"], {}).get("sampled_any10_recovered"),
                           "n_reps": by_id[row["doc_id"]]["n_reps"]})
    # Prefer examples not recovered by V-base, with one example per document.
    traces.sort(key=lambda row: (bool(row["base_greedy_recovered"] or row["base_sampled_any10_recovered"]), row["prompt_id"]))
    distinct = {}
    for trace in traces:
        distinct.setdefault(trace["doc_id"], trace)
    return list(distinct.values())


def collect_part_c(evaluations, results, data, sanitized):
    evaluations, results, data = map(Path, (evaluations, results, data))
    reports = {name: read_json(evaluations / f"partC_{name}_attacks.json")
               for name in ("base", "raw", "san")}
    traces = redact_traces(data, sanitized, reports["san"], reports["base"], reports["raw"])
    combined = {**reports["raw"], "comparisons": reports, "redacted_traces": traces}
    write_json(results / "partC_attacks.json", combined)
    return combined


def plot_part_a(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    results = Path(results)
    with (results / "partA_sweep.csv").open(newline="") as stream:
        sweep = list(csv.DictReader(stream))
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)
    for axis, field, title in zip(axes, ("trainable_pct", "peak_gpu_mem_gb", "train_wallclock_s"),
                                 ("Trainable parameters (%)", "Peak allocated GPU GiB", "Training seconds")):
        for row in sweep:
            x, y = float(row[field]), float(row["val_entity_f1"])
            axis.scatter(x, y)
            offset = {"A4": (-19, 7), "A5": (-25, 4), "A6": (4, -4)}.get(row["run_id"], (3, 3))
            axis.annotate(row["run_id"], (x, y), xytext=offset, textcoords="offset points")
        axis.set(xlabel=title, ylabel="Validation IoU micro-F1")
        axis.grid(alpha=0.2)
    axes[0].set_xscale("log")
    fig.savefig(results / "partA_fig.png", dpi=180)
    plt.close(fig)


def plot_part_c(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    results = Path(results)
    reports = read_json(results / "partC_attacks.json")["comparisons"]
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, axis = plt.subplots(figsize=(6, 3), constrained_layout=True)
    for name, label in (("raw", "V-raw"), ("san", "V-san")):
        points = reports[name]["canary"]["by_n_reps"]
        keys = sorted(map(int, points))
        axis.plot(keys, [points[str(k)]["mean_exposure"] for k in keys], marker="o", label=label)
    axis.set(xlabel="Canary repetitions", ylabel="Mean sampled exposure (bits)", xscale="log")
    axis.set_xticks(keys, labels=[str(k) for k in keys])
    axis.axhline(math.log2(10000), color="gray", linestyle="--", linewidth=0.8, label="Estimator ceiling")
    axis.legend()
    fig.savefig(results / "partC_exposure.png", dpi=180)
    plt.close(fig)
