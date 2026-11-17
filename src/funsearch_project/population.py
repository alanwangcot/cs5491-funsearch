# CS5491: Reasoning Augmented FunSearch
# Author: 朱泳颐 (72657023)
# Responsibility: Search framework and logging

"""Shared deterministic population policy for both representations."""

from .storage import digest


def identity(candidate, mode):
    return digest(
        [
            candidate["code_hash"],
            candidate.get("strategy") if mode == "strategy_code" else None,
        ]
    )


def select_parent(population, rng, tournament_size=3):
    sampled = rng.choices(population, k=tournament_size)
    return min(sampled, key=lambda x: (x["fitness"], x["admission"]))


def admit(population, candidate, fitness, ordinal, mode, maximum=20):
    ident = identity(candidate, mode)
    if any(identity(p["candidate"], mode) == ident for p in population):
        return population, False
    entry = {"candidate": candidate, "fitness": fitness, "admission": ordinal}
    updated = sorted(
        population + [entry], key=lambda x: (x["fitness"], x["admission"])
    )[:maximum]
    return updated, any(p is entry for p in updated)
