# CS5491: Reasoning Augmented FunSearch
# Author: 李逸洋 (72655555)
# Responsibility: Benchmark preparation and evaluation scripts

"""Container entry point. Receives only code, current split, and fixed limits."""

from __future__ import annotations

import json
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from funsearch_project.binpacking import Instance, pack_instance


def run(payload, counters=None):
    counters = counters if counters is not None else {}
    counters.update(instance_executions=0, smoke_executions=0)
    from funsearch_project.validation import compile_candidate

    started = time.monotonic()
    candidate = compile_candidate(payload["code"])
    seconds = float(payload.get("limits", {}).get("instance_seconds", 5))

    def alarm(signum, frame):
        raise TimeoutError("per-instance wall time exceeded")

    signal.signal(signal.SIGALRM, alarm)
    # Small fixed smoke suite precedes full scoring, identical for both arms.
    for index, items in enumerate(
        ((6, 6, 4, 4), (2, 5, 4, 7, 1, 3, 8), (1, 2, 3, 4, 5, 6))
    ):
        counters["smoke_executions"] += 1
        signal.setitimer(signal.ITIMER_REAL, seconds)
        pack_instance(candidate, Instance(f"smoke-{index}", 10, items))
        signal.setitimer(signal.ITIMER_REAL, 0)
    bins = {}
    for raw in payload["dataset"]:
        instance = Instance(**raw)
        counters["instance_executions"] += 1
        signal.setitimer(signal.ITIMER_REAL, seconds)
        bins[instance.instance_id] = pack_instance(
            candidate, instance, diagnostics=False
        ).bins_used
        signal.setitimer(signal.ITIMER_REAL, 0)
    reference = payload["best_fit_bins"]
    ratios = {key: value / reference[key] for key, value in bins.items()}
    return {
        "status": "ok",
        "fitness": sum(ratios.values()) / len(ratios),
        "bins": bins,
        "ratios": ratios,
        "duration_seconds": time.monotonic() - started,
        "error": None,
        "instance_executions": len(bins),
        "smoke_executions": 3,
    }


def main():
    counters = {}
    try:
        result = run(json.load(sys.stdin), counters)
    except Exception as exc:
        result = {
            "status": "timeout" if isinstance(exc, TimeoutError) else "runtime_error",
            "fitness": None,
            "bins": {},
            "ratios": {},
            "error": f"{type(exc).__name__}: {str(exc)[:500]}",
            "instance_executions": counters.get("instance_executions", 0),
            "smoke_executions": counters.get("smoke_executions", 0),
        }
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
