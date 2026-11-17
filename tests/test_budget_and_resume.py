# CS5491: Reasoning Augmented FunSearch
# Author: 朱泳颐 (72657023)
# Responsibility: Search framework and logging

from concurrent.futures import ThreadPoolExecutor

import pytest

from funsearch_project.storage import BudgetExceeded, Ledger


def test_global_reservations_atomic_and_never_recycled(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite", global_cap=9)

    def reserve(index):
        try:
            return ledger.reserve(f"r{index}", "production", 200, {})[0]
        except BudgetExceeded:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = [x for x in pool.map(reserve, range(30)) if x]
    assert len(ids) == len(set(ids)) == 9
    assert ledger.reconciliation()["remaining"] == 0
    with pytest.raises(BudgetExceeded):
        ledger.reserve("new", "production", 1, {})


def test_error_duplicate_retry_all_consume_slots(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite")
    for status in ("transport_error", "invalid_candidate", "duplicate"):
        ident, _ = ledger.reserve("run", "production", 3, {})
        ledger.save_response(ident, {"status": status})
    assert ledger.count("run") == 3
    with pytest.raises(BudgetExceeded):
        ledger.reserve("run", "production", 3, {})
    with pytest.raises(ValueError):
        ledger.save_response(1, {"status": "different"})


def test_development_cap_not_bypassed_by_new_run(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite")
    for index in range(200):
        ledger.reserve(f"d{index}", "development", 1, {})
    with pytest.raises(BudgetExceeded):
        ledger.reserve("another", "development", 10, {})


def test_changed_strategy_is_distinct_candidate():
    from funsearch_project.population import admit

    base = {"code_hash": "same", "strategy": "Fill tightly", "candidate_id": "1"}
    changed = {**base, "strategy": "Preserve gaps", "candidate_id": "2"}
    population, _ = admit([], base, 1.0, 1, "strategy_code")
    population, added = admit(population, changed, 1.0, 2, "strategy_code")
    assert added and len(population) == 2
    population, added = admit(population, changed, 1.0, 3, "strategy_code")
    assert not added and len(population) == 2
