# CS5491: Reasoning Augmented FunSearch
# Author: 李逸洋 (72655555)
# Responsibility: Benchmark preparation and evaluation scripts

"""Re-evaluate the recorded pilot programs without making API calls."""

import json
from pathlib import Path

from funsearch_project.baselines import BEST_FIT_CODE, FIRST_FIT_CODE
from funsearch_project.datasets import load_split
from funsearch_project.evaluator import evaluate_candidate

ROOT = Path(__file__).resolve().parents[1]


def evaluate_pilot(root=ROOT):
    dataset = load_split(root, "search")
    output = {"baselines": {}, "methods": {}}
    cache = {}

    def evaluate(code):
        if code not in cache:
            result = evaluate_candidate(code, dataset)
            if result.status != "ok":
                raise RuntimeError(result.error)
            cache[code] = result
        return cache[code]

    for name, code in [("First Fit", FIRST_FIT_CODE), ("Best Fit", BEST_FIT_CODE)]:
        result = evaluate(code)
        output["baselines"][name] = {
            "total_bins": sum(result.bins.values()),
            "ratio": result.fitness,
        }
    for method in ("code_only", "strategy_code"):
        directory = root / "data/pilot" / method
        scores = []
        for path in sorted((directory / "candidates").glob("c*.json")):
            saved = json.loads(path.read_text())
            result = evaluate(saved["candidate"]["code"])
            assert result.bins == saved["evaluation"]["bins"], path.name
            scores.append(result.fitness)
        output["methods"][method] = {
            "calls": len((directory / "events.jsonl").read_text().splitlines()),
            "valid": len(scores),
            "best_ratio": min(scores),
        }
    return output


if __name__ == "__main__":
    print(json.dumps(evaluate_pilot(), indent=2))
