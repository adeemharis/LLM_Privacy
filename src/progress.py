"""Atomic checkpoints and deadline handling for resumable experiments."""
import json
import os
import time
from pathlib import Path

try:
    from .common import write_json
except ImportError:
    from common import write_json


class BudgetStop(RuntimeError):
    """Signal a time-limited run to stop at a checkpoint boundary."""


def stop_due(now=None):
    deadline = os.environ.get("A1_STOP_AT_UNIX")
    return bool(deadline) and (time.time() if now is None else now) >= float(deadline)


class Journal:
    def __init__(self, path, identity, resume):
        self.path = Path(path)
        self.identity = identity
        self.records, self.elapsed, self.state = [], 0.0, None
        if self.path.exists():
            if not resume:
                raise FileExistsError(f"Progress exists; use --resume: {self.path}")
            previous = json.loads(self.path.read_text())
            if previous["identity"] != identity:
                raise ValueError("Progress fingerprint mismatch; refusing stale output reuse.")
            self.records, self.elapsed = previous["records"], previous["elapsed_s"]
            self.state = previous.get("state")
        self.started = time.perf_counter()

    def save(self):
        write_json(self.path, {"identity": self.identity, "records": self.records,
                              "elapsed_s": self.elapsed + time.perf_counter() - self.started,
                              "state": self.state})

    def check_stop(self):
        if stop_due():
            self.save()
            raise BudgetStop("Time allowance reached; deterministic generation progress saved.")

    @property
    def total_seconds(self):
        return self.elapsed + time.perf_counter() - self.started
