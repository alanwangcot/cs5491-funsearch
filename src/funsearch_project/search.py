# CS5491: Reasoning Augmented FunSearch
# Author: 朱泳颐 (72657023)
# Responsibility: Search framework and logging

"""One representation-controlled search engine, with durable attempt replay."""

from __future__ import annotations

import hashlib
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, is_dataclass
from pathlib import Path

import yaml

from .config import load_config, public_config, require_frozen
from .population import admit, identity, select_parent
from .storage import Ledger, atomic_json, digest, utcnow


def plain(value):
    return asdict(value) if is_dataclass(value) else value


def ledger_for(config):
    return Ledger(
        Path(config["root"]) / "runs/attempts.sqlite3", config.get("global_cap", 2200)
    )


def tuple_tree(value):
    return tuple(tuple_tree(v) for v in value) if isinstance(value, list) else value


def client_for(config):
    from .llm import DeepSeekClient, FakeClient

    provider = config.get("provider", {})
    if provider.get("name") == "fake":
        return FakeClient()
    options = {k: v for k, v in provider.items() if k != "name"}
    if "env_path" in options:
        options["env_path"] = str(
            (Path(config["root"]) / options["env_path"]).resolve()
        )
    return DeepSeekClient(**options)


def evaluation_context(config, dataset):
    import numpy as np

    from .binpacking import WRAPPER_VERSION

    source = Path(__file__).resolve().parent
    code_versions = {
        name: hashlib.sha256((source / name).read_bytes()).hexdigest()
        for name in ("binpacking.py", "evaluator.py", "worker.py", "validation.py")
    }
    return digest(
        {
            "dataset": [plain(i) for i in dataset],
            "limits": config.get("limits", {}),
            "wrapper": WRAPPER_VERSION,
            "code_versions": code_versions,
            "numpy": np.__version__,
        }
    )


def cached_evaluate(code, dataset, config, ledger, evaluator=None):
    from .evaluator import evaluate_candidate

    evaluator = evaluator or evaluate_candidate
    key = digest(
        [hashlib.sha256(code.encode()).hexdigest(), evaluation_context(config, dataset)]
    )
    cached = ledger.cache_get(key)
    if cached is not None:
        return cached, True
    result = plain(evaluator(code, dataset, config.get("limits", {})))
    # Resource failures may be transient. Successful deterministic evaluations alone are cached.
    if result.get("status") in ("ok", "valid", "success"):
        ledger.cache_put(key, result)
    return result, False


def successful(result):
    return (
        result.get("status") in ("ok", "valid", "success")
        and result.get("fitness") is not None
    )


def sync_artifacts(run_dir, ledger, run_id, state):
    atomic_json(run_dir / "checkpoint.json", state)
    rows = ledger.attempts(run_id)
    text = "".join(
        json.dumps(r["event"], sort_keys=True) + "\n" for r in rows if r["event"]
    )
    path = run_dir / "events.jsonl"
    temp = path.with_suffix(".jsonl.tmp")
    temp.write_text(text)
    temp.replace(path)


def run_search(
    config,
    run_id,
    seed,
    resume=False,
    *,
    mode=None,
    phase="production",
    client=None,
    dataset=None,
    evaluator=None,
    stop_after=None,
):
    from .baselines import BEST_FIT_CODE, FIRST_FIT_CODE
    from .candidates import Candidate, parse_candidate
    from .prompts import build_prompt
    from .validation import validate_code

    config = load_config(config)
    root = Path(config["root"])
    if phase == "production" and config.get("provider", {}).get("name") != "fake":
        require_frozen(config)
    mode = mode or ("strategy_code" if "strategy_code" in run_id else "code_only")
    if mode not in ("code_only", "strategy_code"):
        raise ValueError("Unknown representation mode")
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "responses").mkdir(exist_ok=True)
    (run_dir / "candidates").mkdir(exist_ok=True)
    ledger = ledger_for(config)
    if dataset is None:
        from .datasets import load_split

        dataset = load_split(root, "search", benchmark="synthetic")
    context = evaluation_context(config, dataset)
    run_config = {
        "config": public_config(config),
        "run_id": run_id,
        "seed": seed,
        "mode": mode,
        "phase": phase,
        "evaluation_context": context,
    }
    state = ledger.get_state(run_id)
    if state is None:
        if ledger.count(run_id):
            raise RuntimeError("Reservations exist without initial state")
        rng = random.Random(seed)
        population = []
        for index, (code, description) in enumerate(
            (
                (FIRST_FIT_CODE, "Choose the earliest feasible bin."),
                (
                    BEST_FIT_CODE,
                    "Choose the feasible bin leaving the smallest residual capacity.",
                ),
            )
        ):
            code = validate_code(code)
            candidate = Candidate(
                code=code,
                strategy=description if mode == "strategy_code" else None,
                candidate_id=f"seed_{index}",
                code_hash=hashlib.sha256(code.encode()).hexdigest(),
            )
            result, _ = cached_evaluate(code, dataset, config, ledger, evaluator)
            if not successful(result):
                raise RuntimeError(f"Seed evaluation failed: {result}")
            population, _ = admit(
                population,
                plain(candidate),
                result["fitness"],
                index - 2,
                mode,
                config.get("population_size", 20),
            )
            atomic_json(
                run_dir / "candidates" / f"seed_{index}.json",
                {"candidate": plain(candidate), "evaluation": result},
            )
        state = {
            "run_config_hash": digest(run_config),
            "run_id": run_id,
            "seed": seed,
            "mode": mode,
            "phase": phase,
            "population": population,
            "rng_state": rng.getstate(),
            "processed": 0,
            "valid": 0,
            "parsed": 0,
            "unique_full_evaluations": 0,
            "cache_hits": 0,
            "seen_code_hashes": sorted(
                {p["candidate"]["code_hash"] for p in population}
            ),
            "seen_pair_hashes": sorted(
                {identity(p["candidate"], mode) for p in population}
            ),
            "incumbent_curve": [
                {
                    "attempt": 0,
                    "unique_evaluations": 0,
                    "fitness": min(p["fitness"] for p in population),
                }
            ],
            "started_at": utcnow(),
            "status": "running",
        }
        ledger.save_state(run_id, state)
        atomic_json(run_dir / "run_metadata.json", run_config)
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(public_config(config), sort_keys=True)
        )
    elif not resume:
        raise RuntimeError(f"Run {run_id} already exists; resume explicitly")
    if state["run_config_hash"] != digest(run_config):
        raise RuntimeError("Resume configuration/data differs from original run")
    rng = random.Random()
    rng.setstate(tuple_tree(state["rng_state"]))
    client = client or client_for(config)
    consumed_this_call = 0
    cap = config.get("attempts_per_run", 200)
    while True:
        rows = ledger.attempts(run_id)
        pending = [r for r in rows if r["event"] is None]
        if pending:
            row = pending[0]
            metadata = row["metadata"]
            rng.setstate(tuple_tree(metadata["rng_after_parent"]))
            response = row["response"]
            # Unknown dispatch outcome consumes its reserved slot; never resend.
            if response is None:
                response = {
                    "status": "interrupted",
                    "error": "Reserved request had unknown completion at recovery",
                    "raw_text": "",
                }
                ledger.save_response(row["id"], response)
        else:
            if len(rows) >= cap or (
                stop_after is not None and consumed_this_call >= stop_after
            ):
                break
            parent = select_parent(
                state["population"], rng, config.get("tournament_size", 3)
            )
            parent_obj = Candidate(**parent["candidate"])
            messages = build_prompt(parent_obj, mode, parent["fitness"])
            metadata = {
                "parent_id": parent_obj.candidate_id,
                "prompt_hash": digest(messages),
                "messages": messages,
                "model_config": config.get("provider", {}),
                "rng_after_parent": rng.getstate(),
                "mode": mode,
                "seed": seed,
                "is_retry": False,
            }
            attempt_id, ordinal = ledger.reserve(run_id, phase, cap, metadata)
            row = {"id": attempt_id, "ordinal": ordinal, "created": utcnow()}
            started = time.monotonic()
            try:
                response = plain(client.generate(messages))
            except Exception as exc:
                response = {
                    "status": "transport_error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "raw_text": "",
                    "latency": time.monotonic() - started,
                }
            ledger.save_response(attempt_id, response)
        ordinal = row["ordinal"]
        atomic_json(run_dir / "responses" / f"attempt_{ordinal:04d}.json", response)
        event = {
            "global_attempt_id": row["id"],
            "attempt": ordinal,
            "run_id": run_id,
            "mode": mode,
            "seed": seed,
            "phase": phase,
            "created": row["created"],
            "completed": utcnow(),
            "parent_id": metadata["parent_id"],
            "prompt_hash": metadata["prompt_hash"],
            "model_config": metadata["model_config"],
            "response_path": f"responses/attempt_{ordinal:04d}.json",
            "status": response.get("status", "ok"),
            "usage": response.get("usage"),
            "latency": response.get("latency", response.get("latency_seconds")),
            "request_id": response.get("request_id"),
            "response_model": response.get("response_model"),
            "finish_reason": response.get("finish_reason"),
            "cache_hit": False,
            "code_duplicate": False,
            "pair_duplicate": False,
            "admitted": False,
        }
        if response.get("status") in ("ok", "success") and response.get(
            "response_model"
        ):
            previous_model = state.get("response_model")
            if (
                previous_model is not None
                and previous_model != response["response_model"]
            ):
                event["status"] = "model_changed"
                event["error"] = (
                    f"Provider returned model changed from {previous_model} to {response['response_model']}"
                )
                state["status"] = "provider_blocked"
            else:
                state["response_model"] = response["response_model"]
        if (
            response.get("status") in ("ok", "success")
            and event["status"] != "model_changed"
        ):
            try:
                candidate = parse_candidate(response["raw_text"], mode)
                candidate.candidate_id = f"c{ordinal:04d}"
                candidate.code_hash = hashlib.sha256(
                    candidate.code.encode()
                ).hexdigest()
                c = plain(candidate)
                state["parsed"] += 1
                event["code_hash"] = candidate.code_hash
                event["strategy"] = candidate.strategy
                seen_codes = state.setdefault(
                    "seen_code_hashes",
                    sorted({p["candidate"]["code_hash"] for p in state["population"]}),
                )
                seen_pairs = state.setdefault(
                    "seen_pair_hashes",
                    sorted(
                        {identity(p["candidate"], mode) for p in state["population"]}
                    ),
                )
                pair_hash = identity(c, mode)
                event["code_duplicate"] = candidate.code_hash in seen_codes
                event["pair_duplicate"] = pair_hash in seen_pairs
                if not event["code_duplicate"]:
                    seen_codes.append(candidate.code_hash)
                if not event["pair_duplicate"]:
                    seen_pairs.append(pair_hash)
                result, hit = cached_evaluate(
                    candidate.code, dataset, config, ledger, evaluator
                )
                event["cache_hit"] = hit
                event["evaluation"] = result
                event["status"] = result["status"]
                if successful(result):
                    state["valid"] += 1
                    state["cache_hits"] += int(hit)
                    state["unique_full_evaluations"] += int(not hit)
                    state["population"], event["admitted"] = admit(
                        state["population"],
                        c,
                        result["fitness"],
                        ordinal,
                        mode,
                        config.get("population_size", 20),
                    )
                    event["fitness"] = result["fitness"]
                atomic_json(
                    run_dir / "candidates" / f"c{ordinal:04d}.json",
                    {"candidate": c, "evaluation": result},
                )
            except (ValueError, SyntaxError, TypeError, KeyError) as exc:
                event["status"] = "invalid_candidate"
                event["error"] = f"{type(exc).__name__}: {exc}"
        elif event["status"] != "model_changed":
            event["error"] = response.get("error")
        incumbent = min(
            state["population"], key=lambda p: (p["fitness"], p["admission"])
        )
        event["incumbent_id"] = incumbent["candidate"]["candidate_id"]
        event["incumbent_fitness"] = incumbent["fitness"]
        event["unique_full_evaluations"] = state["unique_full_evaluations"]
        state["processed"] = ordinal
        state["rng_state"] = rng.getstate()
        state["incumbent_curve"].append(
            {
                "attempt": ordinal,
                "unique_evaluations": state["unique_full_evaluations"],
                "fitness": incumbent["fitness"],
            }
        )
        ledger.finalize(row["id"], event, state)
        sync_artifacts(run_dir, ledger, run_id, state)
        consumed_this_call += 1
        if response.get("status") == "fatal" or event["status"] == "model_changed":
            state["status"] = "provider_blocked"
            break
    if state.get("status") != "provider_blocked":
        state["status"] = "complete" if ledger.count(run_id) >= cap else "partial"
    state["updated_at"] = utcnow()
    ledger.save_state(run_id, state)
    sync_artifacts(run_dir, ledger, run_id, state)
    summary = {
        k: v
        for k, v in state.items()
        if k
        not in {
            "rng_state",
            "population",
            "incumbent_curve",
            "seen_code_hashes",
            "seen_pair_hashes",
        }
    }
    summary.update(
        {
            "attempts": ledger.count(run_id),
            "best_search_fitness": min(p["fitness"] for p in state["population"]),
            "retained": len(state["population"]),
        }
    )
    atomic_json(run_dir / "summary.json", summary)
    return summary


def run_all(config, phase="production"):
    config = load_config(config)
    if phase == "production":
        require_frozen(config)
        from .config import verify_runtime

        verify_runtime(config)
    ledger = ledger_for(config)
    jobs = []
    for index, seed in enumerate(config["seeds"]):
        modes = config.get("modes", ["code_only", "strategy_code"])
        for mode in modes if index % 2 == 0 else list(reversed(modes)):
            run_id = f"{config.get('experiment_id', 'main')}_{mode}_{seed}"
            jobs.append((run_id, seed, mode))
    remaining_required = sum(
        max(0, config["attempts_per_run"] - ledger.count(run_id))
        for run_id, _, _ in jobs
    )
    if ledger.count() + remaining_required > config.get("global_cap", 2200):
        raise RuntimeError("Insufficient reserved-attempt budget for configured runs")

    def execute(job):
        run_id, seed, mode = job
        return run_search(
            config,
            run_id,
            seed,
            resume=ledger.get_state(run_id) is not None,
            mode=mode,
            phase=phase,
        )

    with ThreadPoolExecutor(max_workers=config.get("workers", 1)) as pool:
        results = list(pool.map(execute, jobs))
    atomic_json(
        Path(config["root"]) / "results/budget_reconciliation.json",
        ledger.reconciliation(),
    )
    return results
