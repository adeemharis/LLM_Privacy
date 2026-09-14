"""Regex and default Presidio baselines for PII masking."""
import ipaddress
import re

RULES = [
    ("EMAIL", r"(?<![\w.+-])[\w.!#$%&'*+/=?^`{|}~-]+@[\w-]+(?:\.[\w-]+)+", 100),
    ("URL", r"\b(?:https?://|www\.)[^\s<>\[\]]+", 95),
    ("IP", r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?!\d|\.\d)", 90),
    ("IP", r"(?<![\w:])(?:[0-9a-fA-F]{0,4}:){2,7}[0-9a-fA-F]{0,4}(?![\w:])", 90),
    ("SOCIALNUMBER", r"\b\d{3}-\d{2}-\d{4}\b", 85),
    ("CREDITCARDNUMBER", r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)", 80),
    ("DATE", r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b", 75),
    ("DATE", r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?\b", 75),
    ("TIME", r"\b(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?(?:\s*[ap]\.?m\.?)?\b|\b(?:0?[1-9]|1[0-2])\s*[ap]m\b", 70),
    ("TEL", r"(?<!\w)(?:\+\d{1,3}[ .-]?)?(?:\(\d{2,4}\)|\d{2,4})[ .-]\d{3,4}[ .-]\d{3,4}(?!\w)", 60),
]


def luhn(value):
    digits = [int(c) for c in value if c.isdigit() and c.isascii()]
    if not 13 <= len(digits) <= 19:
        return False
    return sum((d * 2 - 9 if d * 2 > 9 else d * 2) if i % 2 else d
               for i, d in enumerate(reversed(digits))) % 10 == 0


def replace_spans(source, spans):
    parts, cursor = [], 0
    for span in sorted(spans, key=lambda s: s["start"]):
        parts.extend((source[cursor:span["start"]], f"[{span['label']}]"))
        cursor = span["end"]
    return "".join(parts) + source[cursor:]


def regex_sanitize(source):
    candidates = []
    for label, pattern, priority in RULES:
        for match in re.finditer(pattern, source, re.IGNORECASE):
            start, end = match.span()
            if label == "URL":
                end = start + len(match.group().rstrip(".,;:!?)}"))
            if label == "CREDITCARDNUMBER" and not luhn(match.group()):
                continue
            if label == "IP":
                try:
                    ipaddress.ip_address(match.group())
                except ValueError:
                    continue
            candidates.append((priority, start, end, label))
    selected = []
    for _, start, end, label in sorted(candidates, key=lambda c: (-c[0], -(c[2] - c[1]), c[1])):
        if not any(start < s["end"] and s["start"] < end for s in selected):
            selected.append({"start": start, "end": end, "label": label})
    selected.sort(key=lambda s: s["start"])
    return replace_spans(source, selected), selected


class PresidioBaseline:
    def __init__(self):
        from presidio_analyzer import AnalyzerEngine
        self.engine = AnalyzerEngine()

    def __call__(self, source):
        # Map equivalent labels only. PERSON and LOCATION stay coarse because
        # the recognizers do not distinguish the dataset's finer classes.
        aliases = {"EMAIL_ADDRESS": "EMAIL", "PHONE_NUMBER": "TEL", "IP_ADDRESS": "IP",
                   "US_SSN": "SOCIALNUMBER", "CREDIT_CARD": "CREDITCARDNUMBER"}
        selected = []
        for item in sorted(self.engine.analyze(text=source, language="en"),
                           key=lambda x: (-x.score, -(x.end - x.start), x.start)):
            if not any(item.start < s["end"] and s["start"] < item.end for s in selected):
                selected.append({"start": item.start, "end": item.end,
                                 "label": aliases.get(item.entity_type, item.entity_type)})
        selected.sort(key=lambda s: s["start"])
        return replace_spans(source, selected), selected
