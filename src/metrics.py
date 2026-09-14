"""Character-offset alignment, entity matching and sanitization metrics."""
import re
from collections import Counter
from difflib import SequenceMatcher
from functools import lru_cache

try:
    from .common import contains_value
except ImportError:
    from common import contains_value

MASK = re.compile(r"\[([A-Z][A-Z0-9_]*?)\]")


def canonical_label(label):
    # GIVENNAME1 is an actual class. Only an underscore-delimited counter is removed.
    return re.sub(r"_\d+$", "", label)


def recover_spans(source, output):
    """Use unchanged text as anchors; collapse new masks before character alignment.

    Literal masks copied from the input are protected. A changed source region
    with exactly one marker is a prediction. Adjacent markers without an
    unambiguous literal boundary are not arbitrarily split. Their count is
    reported and they contribute unmatched predictions in the evaluator.
    """
    equals = SequenceMatcher(None, source, output, autojunk=False).get_matching_blocks()
    sentinel = next(chr(c) for c in range(0xE000, 0xF8FF)
                    if chr(c) not in source and chr(c) not in output)
    pieces, markers, cursor, length = [], {}, 0, 0
    for match in MASK.finditer(output):
        literal = output[cursor:match.start()]
        pieces.append(literal)
        length += len(literal)
        copied = any(b <= match.start() and match.end() <= b + n for _, b, n in equals)
        if copied:
            pieces.append(match.group())
            length += len(match.group())
        else:
            pieces.append(sentinel)
            markers[length] = canonical_label(match.group(1))
            length += 1
        cursor = match.end()
    pieces.append(output[cursor:])
    skeleton = "".join(pieces)
    # First try exact literal anchoring over the entire document. This prevents
    # SequenceMatcher from aligning terminal punctuation to punctuation inside
    # a removed email or address.
    literals = skeleton.split(sentinel)
    ordered_labels = list(markers.values())
    if markers and all(literals[i] for i in range(1, len(literals) - 1)) and source.startswith(literals[0]):
        @lru_cache(maxsize=None)
        def anchor(index, position):
            literal = literals[index + 1]
            if index == len(ordered_labels) - 1:
                boundary = len(source) - len(literal)
                if boundary > position and source[boundary:] == literal:
                    return ((position, boundary),)
                return None
            boundary = source.find(literal, position + 1)
            while boundary >= 0:
                rest = anchor(index + 1, boundary + len(literal))
                if rest is not None:
                    return ((position, boundary),) + rest
                boundary = source.find(literal, boundary + 1)
            return None
        anchored = anchor(0, len(literals[0]))
        if anchored is not None:
            spans = [{"start": a, "end": b, "label": label}
                     for (a, b), label in zip(anchored, ordered_labels)]
            return {"spans": spans, "changed_ranges": list(anchored), "alignment_errors": {},
                    "unmatched_prediction_labels": [], "well_formed": True}
    spans, changed, errors, unmatched = [], [], Counter(), []
    for tag, a0, a1, b0, b1 in SequenceMatcher(
            None, source, skeleton, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if a0 != a1:
            changed.append((a0, a1))
        labels = [label for pos, label in markers.items() if b0 <= pos < b1]
        residue = skeleton[b0:b1].replace(sentinel, "")
        if len(labels) == 1 and a0 < a1 and not residue.strip():
            spans.append({"start": a0, "end": a1, "label": labels[0]})
            if residue:
                errors["whitespace_rewrite"] += 1
        else:
            unmatched.extend(labels)
            if len(labels) > 1:
                errors["ambiguous_adjacent_masks"] += 1
            elif labels:
                errors["unanchored_mask"] += 1
            if residue or (not labels and a0 != a1):
                errors["non_mask_rewrite"] += 1
    return {"spans": spans, "changed_ranges": changed, "alignment_errors": dict(errors),
            "unmatched_prediction_labels": unmatched, "well_formed": not errors}


def iou(a, b):
    intersection = max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    union = max(a["end"], b["end"]) - min(a["start"], b["start"])
    return intersection / union if union > 0 else 0.0


def maximum_weight_pairs(weights):
    """Hungarian assignment with zero-weight dummy vertices; maximize total IoU."""
    if not weights or not weights[0]:
        return []
    nrows, ncols = len(weights), len(weights[0])
    n = max(nrows, ncols)
    costs = [[-(weights[i][j] if i < nrows and j < ncols else 0.0)
              for j in range(n)] for i in range(n)]
    u, v, p, way = [0.0] * (n + 1), [0.0] * (n + 1), [0] * (n + 1), [0] * (n + 1)
    for row in range(1, n + 1):
        p[0] = row
        column = 0
        minimum = [float("inf")] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[column] = True
            active_row, delta, next_column = p[column], float("inf"), 0
            for j in range(1, n + 1):
                if not used[j]:
                    reduced = costs[active_row - 1][j - 1] - u[active_row] - v[j]
                    if reduced < minimum[j]:
                        minimum[j], way[j] = reduced, column
                    if minimum[j] < delta:
                        delta, next_column = minimum[j], j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            column = next_column
            if p[column] == 0:
                break
        while column:
            previous = way[column]
            p[column] = p[previous]
            column = previous
    return [(p[j] - 1, j - 1) for j in range(1, n + 1)
            if 0 < p[j] <= nrows and j <= ncols and weights[p[j] - 1][j - 1] > 0]


def match_entities(predicted, gold, partial=False, label_match=True):
    if not predicted or not gold:
        return []
    weights = []
    for pred in predicted:
        row = []
        for target in gold:
            weight = iou(pred, target) if partial else float(
                pred["start"] == target["start"] and pred["end"] == target["end"])
            if weight < (0.5 if partial else 1.0):
                weight = 0.0
            if label_match and pred["label"] != target["label"]:
                weight = 0.0
            row.append(weight)
        weights.append(row)
    return maximum_weight_pairs(weights)


def prf(tp, n_pred, n_gold):
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    return {"p": precision, "r": recall,
            "f1": 2 * tp / (n_pred + n_gold) if n_pred + n_gold else 0.0,
            "support": n_gold, "predicted": n_pred, "tp": tp}


class Metrics:
    def __init__(self):
        self.n = self.well_formed = self.leaked = self.gold_values = 0
        self.non_pii = self.overmasked = 0
        self.gold, self.pred = Counter(), Counter()
        self.tp = {"exact": Counter(), "overlap_iou50": Counter()}
        self.errors, self.alignment = Counter(), Counter()
        self.label_agnostic_tp = 0

    def add(self, row, output, direct_spans=None, latency_ms=0.0):
        source, gold = row["source_text"], row["privacy_mask"]
        recovered = recover_spans(source, output)
        predicted = recovered["spans"] if direct_spans is None else direct_spans
        unmatched = recovered["unmatched_prediction_labels"] if direct_spans is None else []
        self.n += 1
        self.well_formed += int(recovered["well_formed"] if direct_spans is None else True)
        self.gold.update(g["label"] for g in gold)
        self.pred.update(p["label"] for p in predicted)
        self.pred.update(unmatched)
        self.alignment.update(recovered["alignment_errors"])
        for name, partial in (("exact", False), ("overlap_iou50", True)):
            pairs = match_entities(predicted, gold, partial)
            self.tp[name].update(gold[j]["label"] for _, j in pairs)
        exact_gold = {j for _, j in match_entities(predicted, gold)}
        overlap_gold = {j for _, j in match_entities(predicted, gold, True)}
        agnostic = match_entities(predicted, gold, True, label_match=False)
        self.label_agnostic_tp += len(agnostic)
        any_label_gold = {j for _, j in agnostic}
        # Mutually exclusive taxonomy over gold entities that missed exact matching.
        for j, entity in enumerate(gold):
            if j in exact_gold:
                continue
            if j in overlap_gold:
                category = "boundary_error"
            elif j in any_label_gold:
                category = "label_error"
            elif contains_value(output, entity["value"]):
                category = "verbatim_leak"
            else:
                category = "unrecoverable_or_incorrect_replacement"
            self.errors[category] += 1
        leaked_values = [g["value"] for g in gold if contains_value(output, g["value"])]
        self.leaked += len(leaked_values)
        self.gold_values += len(gold)
        pii = bytearray(len(source))
        for entity in gold:
            pii[entity["start"]:entity["end"]] = b"\x01" * (entity["end"] - entity["start"])
        self.non_pii += len(source) - sum(pii)
        changed_ranges = recovered["changed_ranges"] if direct_spans is None else [
            (span["start"], span["end"]) for span in direct_spans]
        changed_indices = {i for a, b in changed_ranges for i in range(a, b)}
        self.overmasked += sum(1 - pii[i] for i in changed_indices)
        return {"id": row["id"], "language": row.get("language", "en"),
                "source_text": source, "predicted_text": output, "predicted_spans": predicted,
                "gold_spans": [{k: g[k] for k in ("start", "end", "label")} for g in gold],
                "leaked_values": leaked_values, "latency_ms": latency_ms,
                "alignment_errors": recovered["alignment_errors"],
                "unmatched_prediction_labels": unmatched}

    def report(self):
        result = {"n": self.n, "leak_rate": self.leaked / self.gold_values if self.gold_values else 0.0,
                  "over_masking_rate": self.overmasked / self.non_pii if self.non_pii else 0.0,
                  "well_formed_rate": self.well_formed / self.n if self.n else 0.0,
                  "counts": {"gold_values": self.gold_values, "leaked_values": self.leaked,
                             "non_pii_characters": self.non_pii, "overmasked_characters": self.overmasked},
                  "error_taxonomy": dict(self.errors), "alignment_errors": dict(self.alignment),
                  "macro_class_policy": "gold-supported labels; unknown predictions still count in micro precision"}
        classes = sorted(self.gold)
        for name, counts in self.tp.items():
            per_class = {label: prf(counts[label], self.pred[label], self.gold[label])
                         for label in sorted(self.gold.keys() | self.pred.keys())}
            micro = prf(sum(counts.values()), sum(self.pred.values()), sum(self.gold.values()))
            result[name] = {"micro_p": micro["p"], "micro_r": micro["r"], "micro_f1": micro["f1"],
                            **{f"macro_{metric}": sum(per_class[c][metric] for c in classes) / len(classes)
                               if classes else 0.0 for metric in ("p", "r", "f1")}}
            result[f"per_class_{name}"] = per_class
        result["per_class"] = result["per_class_exact"]
        result["top10_classes"] = [label for label, _ in self.gold.most_common(10)]
        result["label_agnostic_overlap_diagnostic"] = prf(
            self.label_agnostic_tp, sum(self.pred.values()), sum(self.gold.values()))
        return result
