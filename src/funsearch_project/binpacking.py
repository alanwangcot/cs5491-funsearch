# CS5491: Reasoning Augmented FunSearch
# Author: 李逸洋 (72655555)
# Responsibility: Benchmark preparation and evaluation scripts

"""Fixed online packing wrapper; no future items are exposed to heuristics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

WRAPPER_VERSION = "online-feasible-only-integer-v1"


@dataclass(frozen=True)
class Instance:
    instance_id: str
    capacity: int | float
    items: tuple[int | float, ...]
    split: str = "development"
    source_id: str | None = None
    order: int = 0
    benchmark: str = "synthetic"

    def __post_init__(self):
        object.__setattr__(self, "items", tuple(self.items))
        if not math.isfinite(self.capacity) or self.capacity <= 0:
            raise ValueError("capacity must be positive and finite")
        if any(not math.isfinite(x) or not 0 < x <= self.capacity for x in self.items):
            raise ValueError("items must be finite, positive, and fit capacity")


@dataclass
class PackingResult:
    bins_used: int
    remaining: list[int | float]
    placements: list[int] = field(default_factory=list)
    valid: bool = True


def pack_instance(
    candidate: Callable, instance: Instance, *, diagnostics: bool = True
) -> PackingResult:
    """Maximize finite scores over feasible opened bins; tie by opening order."""
    integer = isinstance(instance.capacity, int) and all(
        isinstance(x, int) for x in instance.items
    )
    dtype = np.int64 if integer else np.float64
    remaining = np.empty(len(instance.items), dtype=dtype)
    opened = 0
    placements = []
    total = 0
    for item in instance.items:
        active = remaining[:opened]
        indices = np.flatnonzero(active >= item)
        if indices.size:
            capacities = active[indices].copy()
            original = capacities.copy()
            scores = np.asarray(candidate(item, capacities))
            if not np.array_equal(capacities, original):
                raise ValueError("mutation: candidate changed remaining capacities")
            if (
                scores.ndim != 1
                or scores.shape != capacities.shape
                or scores.dtype.kind not in "iuf"
            ):
                raise ValueError(
                    "shape: score must be one numeric value per feasible bin"
                )
            if not np.all(np.isfinite(scores)):
                raise ValueError("nonfinite: scores must be finite")
            chosen = int(indices[int(np.argmax(scores))])
            remaining[chosen] -= item
        else:
            chosen = opened
            remaining[opened] = instance.capacity - item
            opened += 1
        total += item
        if remaining[chosen] < 0 or remaining[chosen] > instance.capacity:
            raise ValueError("capacity invariant violated")
        # Running integer conservation is exact, avoiding a full sum at each placement.
        if diagnostics:
            placements.append(chosen)
            residual = float(np.sum(remaining[:opened]))
            if not math.isclose(
                residual + total,
                opened * instance.capacity,
                rel_tol=1e-12,
                abs_tol=1e-8,
            ):
                raise ValueError("capacity conservation violated")
    if not math.isclose(
        float(np.sum(remaining[:opened])) + total,
        opened * instance.capacity,
        rel_tol=1e-12,
        abs_tol=1e-8,
    ):
        raise ValueError("final capacity conservation violated")
    return PackingResult(opened, remaining[:opened].tolist(), placements)
