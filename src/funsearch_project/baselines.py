# CS5491: Reasoning Augmented FunSearch
# Author: 李逸洋 (72655555)
# Responsibility: Benchmark preparation and evaluation scripts

"""Classical algorithms use precisely the same online wrapper."""

import numpy as np

FIRST_FIT_CODE = (
    "def score_bins(item, remaining):\n    return np.zeros_like(remaining)\n"
)
BEST_FIT_CODE = "def score_bins(item, remaining):\n    return -(remaining - item)\n"
FIRST_FIT_STRATEGY = "Choose the earliest opened feasible bin."
BEST_FIT_STRATEGY = "Choose the feasible bin leaving the smallest residual capacity."


def first_fit(item, remaining):
    return np.zeros_like(remaining)


def best_fit(item, remaining):
    return -(remaining - item)
