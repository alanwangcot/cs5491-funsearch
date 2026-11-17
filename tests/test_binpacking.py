# CS5491: Reasoning Augmented FunSearch
# Author: 李逸洋 (72655555)
# Responsibility: Benchmark preparation and evaluation scripts

import numpy as np
import pytest

from funsearch_project.baselines import best_fit, first_fit
from funsearch_project.binpacking import Instance, pack_instance


@pytest.mark.parametrize("candidate", [first_fit, best_fit])
@pytest.mark.parametrize(
    "items,expected",
    [
        ((), 0),
        ((6, 6, 4, 4), 2),
        ((2, 5, 4, 7, 1, 3, 8), 4),
        ((10, 10, 10), 3),
        ((5, 5, 5, 5), 2),
    ],
)
def test_hand_examples(candidate, items, expected):
    result = pack_instance(candidate, Instance("fixture", 10, items))
    assert result.bins_used == expected
    assert sum(result.remaining) + sum(items) == 10 * expected


def test_earliest_tie_and_current_information_only():
    seen = []

    def candidate(item, remaining):
        seen.append((item, remaining.copy()))
        return np.zeros_like(remaining)

    result = pack_instance(candidate, Instance("fixture", 10, (6, 6, 4, 4)))
    assert result.placements == [0, 1, 0, 1]
    assert seen[0][0] == 4
    assert seen[0][1].tolist() == [4, 4]
    assert len(seen) == 2


@pytest.mark.parametrize(
    "candidate",
    [
        lambda i, r: 1,
        lambda i, r: np.zeros((len(r), 1)),
        lambda i, r: np.full_like(r, np.nan, dtype=float),
        lambda i, r: np.full_like(r, np.inf, dtype=float),
        lambda i, r: np.array(["bad"] * len(r)),
    ],
)
def test_bad_scores(candidate):
    with pytest.raises(ValueError):
        pack_instance(candidate, Instance("fixture", 10, (6, 4)))


def test_reject_input_mutation():
    def candidate(i, r):
        r[:] = 0
        return np.zeros_like(r)

    with pytest.raises(ValueError, match="mutation"):
        pack_instance(candidate, Instance("fixture", 10, (6, 4)))


@pytest.mark.parametrize(
    "capacity,items", [(0, (1,)), (10, (0,)), (10, (11,)), (10, (float("nan"),))]
)
def test_invalid_instance(capacity, items):
    with pytest.raises(ValueError):
        Instance("bad", capacity, items)
