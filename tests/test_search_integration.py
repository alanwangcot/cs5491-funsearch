# CS5491: Reasoning Augmented FunSearch
# Author: 朱泳颐 (72657023)
# Responsibility: Search framework and logging

import json

from funsearch_project.binpacking import Instance
from funsearch_project.llm import FakeClient, GenerationResult
from funsearch_project.search import ledger_for, run_search

CODE = "def score_bins(item, remaining):\n    return -(remaining - item)\n"


def fixture_evaluator(code, dataset, limits):
    # Controlled trusted fixture; tests focus on orchestration, not generated execution.
    return {
        "status": "ok",
        "fitness": 1.0 if "zeros" not in code else 1.1,
        "bins": {i.instance_id: 2 for i in dataset},
        "duration_seconds": 0,
    }


def setup(root):
    return {
        "root": str(root),
        "attempts_per_run": 5,
        "global_cap": 2200,
        "provider": {"name": "fake"},
        "limits": {},
        "population_size": 20,
        "tournament_size": 3,
    }, [Instance("fixture", 10, (6, 6, 4, 4))]


def client(mode):
    value = {"code": CODE}
    if mode == "strategy_code":
        value["strategy"] = "Fill the tightest feasible bin."
    return FakeClient([json.dumps(value)])


def test_both_modes_complete_offline_and_failed_attempts_flat(tmp_path):
    cfg, dataset = setup(tmp_path)
    for mode in ("code_only", "strategy_code"):
        provider = client(mode)
        summary = run_search(
            cfg,
            mode,
            11,
            mode=mode,
            client=provider,
            dataset=dataset,
            evaluator=fixture_evaluator,
        )
        assert (
            summary["status"] == "complete"
            and summary["attempts"] == provider.calls == 5
        )
        events = ledger_for(cfg).attempts(mode)
        assert all(r["event"]["cache_hit"] for r in events)
    cfg["attempts_per_run"] = 3
    bad = FakeClient(
        [GenerationResult(status="http_error"), "not json", json.dumps({"code": CODE})]
    )
    summary = run_search(
        cfg, "failed", 22, client=bad, dataset=dataset, evaluator=fixture_evaluator
    )
    state = ledger_for(cfg).get_state("failed")
    assert summary["attempts"] == 3 and summary["parsed"] == 1
    assert all(p["fitness"] == 1.0 for p in state["incumbent_curve"])


def test_resume_preserves_population_rng_and_budget(tmp_path):
    full, dataset = setup(tmp_path / "full")
    resumed, _ = setup(tmp_path / "resumed")
    run_search(
        full,
        "run",
        33,
        client=client("strategy_code"),
        mode="strategy_code",
        dataset=dataset,
        evaluator=fixture_evaluator,
    )
    first_client = client("strategy_code")
    run_search(
        resumed,
        "run",
        33,
        client=first_client,
        mode="strategy_code",
        dataset=dataset,
        evaluator=fixture_evaluator,
        stop_after=2,
    )
    second_client = client("strategy_code")
    run_search(
        resumed,
        "run",
        33,
        resume=True,
        client=second_client,
        mode="strategy_code",
        dataset=dataset,
        evaluator=fixture_evaluator,
    )
    one = ledger_for(full).get_state("run")
    two = ledger_for(resumed).get_state("run")
    for key in (
        "population",
        "rng_state",
        "processed",
        "parsed",
        "valid",
        "incumbent_curve",
    ):
        assert one[key] == two[key]
    assert first_client.calls + second_client.calls == 5


def test_unknown_reserved_attempt_is_not_resent(tmp_path):
    cfg, dataset = setup(tmp_path)
    run_search(
        cfg,
        "run",
        44,
        client=client("code_only"),
        dataset=dataset,
        evaluator=fixture_evaluator,
        stop_after=1,
    )
    ledger = ledger_for(cfg)
    state = ledger.get_state("run")
    previous = ledger.attempts("run")[0]["metadata"]
    ledger.reserve(
        "run", "production", 5, {**previous, "rng_after_parent": state["rng_state"]}
    )
    provider = client("code_only")
    run_search(
        cfg,
        "run",
        44,
        resume=True,
        client=provider,
        dataset=dataset,
        evaluator=fixture_evaluator,
    )
    assert ledger.count("run") == 5 and provider.calls == 3
    assert ledger.attempts("run")[1]["event"]["status"] == "interrupted"


def test_freeze_refuses_changed_prompts(tmp_path):
    import pytest

    from funsearch_project.config import freeze_protocol, require_frozen

    cfg, _ = setup(tmp_path)
    (tmp_path / "prompts").mkdir()
    prompt = tmp_path / "prompts/common.txt"
    prompt.write_text("original")
    freeze_protocol(cfg)
    require_frozen(cfg)
    prompt.write_text("changed")
    with pytest.raises(RuntimeError, match="changed after freeze"):
        require_frozen(cfg)


def test_saved_response_replayed_after_processing_crash_without_api_resend(tmp_path):
    import pytest

    cfg, dataset = setup(tmp_path)
    novel = "def score_bins(item, remaining):\n    return -2 * (remaining - item)\n"
    provider = FakeClient([json.dumps({"code": novel})])

    def interrupted_evaluator(code, items, limits):
        if "-2 *" in code:
            raise RuntimeError("forced processing interruption")
        return fixture_evaluator(code, items, limits)

    with pytest.raises(RuntimeError, match="forced processing"):
        run_search(
            cfg,
            "run",
            55,
            client=provider,
            dataset=dataset,
            evaluator=interrupted_evaluator,
        )
    ledger = ledger_for(cfg)
    assert provider.calls == ledger.count("run") == 1
    assert ledger.attempts("run")[0]["response"] is not None
    assert ledger.attempts("run")[0]["event"] is None
    replay_client = client("code_only")
    run_search(
        cfg,
        "run",
        55,
        resume=True,
        client=replay_client,
        dataset=dataset,
        evaluator=fixture_evaluator,
    )
    assert replay_client.calls == 4 and ledger.count("run") == 5
    assert ledger.get_state("run")["processed"] == 5


def test_provider_model_change_blocks_combining_run(tmp_path):
    cfg, dataset = setup(tmp_path)
    provider = FakeClient(
        [
            GenerationResult(
                raw_text=json.dumps({"code": CODE}), response_model="version-a"
            ),
            GenerationResult(
                raw_text=json.dumps({"code": CODE}), response_model="version-b"
            ),
        ]
    )
    result = run_search(
        cfg, "run", 22, client=provider, dataset=dataset, evaluator=fixture_evaluator
    )
    assert result["status"] == "provider_blocked" and result["attempts"] == 2
    assert ledger_for(cfg).attempts("run")[-1]["event"]["status"] == "model_changed"
