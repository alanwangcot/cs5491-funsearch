# CS5491: Reasoning Augmented FunSearch
# Author: 朱泳颐 (72657023)
# Responsibility: Search framework and logging

"""Commands for dataset preparation, baseline evaluation and program search."""

import argparse
import json
from pathlib import Path

from .config import load_config
from .storage import atomic_json


def prepare(config):
    from .datasets import prepare_data

    return prepare_data(Path(config["root"]))


def baselines(config):
    from .baselines import BEST_FIT_CODE, FIRST_FIT_CODE
    from .datasets import load_split
    from .search import cached_evaluate, ledger_for

    ledger = ledger_for(config)
    rows = []
    for benchmark in ("synthetic", "or_library"):
        for split in ("search", "validation"):
            dataset = load_split(config["root"], split, benchmark=benchmark)
            if not dataset:
                continue
            for name, code in (
                ("First Fit", FIRST_FIT_CODE),
                ("Best Fit", BEST_FIT_CODE),
            ):
                result, _ = cached_evaluate(code, dataset, config, ledger)
                rows.append(
                    {"benchmark": benchmark, "split": split, "method": name, **result}
                )
    atomic_json(Path(config["root"]) / "results/baselines.json", rows)
    return rows


def smoke(config):
    from tempfile import TemporaryDirectory

    from .binpacking import Instance
    from .llm import FakeClient
    from .search import run_search

    with TemporaryDirectory(prefix="funsearch-smoke-") as directory:
        cfg = {
            **config,
            "root": directory,
            "attempts_per_run": 3,
            "provider": {"name": "fake"},
        }
        dataset = [Instance("smoke", 10, (6, 6, 4, 4))]
        return [
            run_search(
                cfg,
                f"smoke_{mode}",
                17,
                mode=mode,
                phase="development",
                client=FakeClient(),
                dataset=dataset,
            )
            for mode in ("code_only", "strategy_code")
        ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["data", "baselines", "smoke", "pilot", "resume"]
    )
    parser.add_argument("action", nargs="?")
    parser.add_argument("--config", default="configs/development.yaml")
    parser.add_argument("--run-id")
    parser.add_argument("--provider", choices=["fake", "deepseek"])
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.provider:
        config["provider"]["name"] = args.provider
    from .search import ledger_for, run_all, run_search

    if args.command == "data":
        if args.action != "prepare":
            parser.error("data requires prepare")
        result = prepare(config)
    elif args.command == "baselines":
        result = baselines(config)
    elif args.command == "smoke":
        result = smoke(config)
    elif args.command == "pilot":
        result = run_all(config, phase="development")
    else:
        if not args.run_id:
            parser.error("resume requires --run-id")
        state = ledger_for(config).get_state(args.run_id)
        if state is None:
            parser.error("unknown run ID")
        result = run_search(
            config,
            args.run_id,
            state["seed"],
            resume=True,
            mode=state["mode"],
            phase=state["phase"],
        )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
