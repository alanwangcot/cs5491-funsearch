# CS5491: Reasoning Augmented FunSearch
# Author: 王宇浩 (72655080)
# Responsibility: LLM interface and strategy augmentation

"""One shared problem prompt; representation is the only treatment difference."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .candidates import Candidate
from .validation import SAFE_NUMPY_NAMES, validate_code

PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"


def prompt_hashes() -> dict[str, str]:
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(PROMPT_DIR.glob("*.txt"))
    }


def build_prompt(
    parent: Candidate, mode: str, parent_fitness: float
) -> list[dict[str, str]]:
    if mode not in {"code_only", "strategy_code"}:
        raise ValueError("unknown representation mode")
    common = (
        (PROMPT_DIR / "common.txt")
        .read_text()
        .replace("{numpy_names}", ", ".join(sorted(SAFE_NUMPY_NAMES)))
    )
    template = (PROMPT_DIR / f"{mode}.txt").read_text()
    # Explicit replacement avoids interpreting the literal JSON example braces.
    content = template.replace("{fitness}", format(parent_fitness, ".12g"))
    content = content.replace("{code}", validate_code(parent.code))
    if mode == "strategy_code":
        content = content.replace(
            "{strategy}",
            parent.strategy or "Choose the earliest feasible bin when scores tie.",
        )
    return [{"role": "system", "content": common}, {"role": "user", "content": content}]
