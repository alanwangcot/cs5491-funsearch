# CS5491: Reasoning Augmented FunSearch
# Author: 王宇浩 (72655080)
# Responsibility: LLM interface and strategy augmentation

"""Auditable response parsing and candidate identities."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from .validation import ValidationError, validate_code


class CandidateError(ValueError):
    pass


@dataclass
class Candidate:
    code: str
    strategy: str | None = None
    candidate_id: str = ""
    code_hash: str = ""
    raw_code: str = ""

    def __post_init__(self):
        if not self.code_hash:
            self.code_hash = hashlib.sha256(self.code.encode()).hexdigest()
        if not self.raw_code:
            self.raw_code = self.code

    @property
    def strategy_hash(self) -> str | None:
        return (
            hashlib.sha256(self.strategy.encode()).hexdigest()
            if self.strategy is not None
            else None
        )

    @property
    def identity(self) -> str:
        return hashlib.sha256(
            json.dumps([self.code_hash, self.strategy], ensure_ascii=True).encode()
        ).hexdigest()


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CandidateError("duplicate JSON field")
        result[key] = value
    return result


def parse_candidate(raw: str, mode: str) -> Candidate:
    if mode not in {"code_only", "strategy_code"}:
        raise CandidateError("unknown representation mode")
    if not isinstance(raw, str) or len(raw) > 50000:
        raise CandidateError("invalid response size/type")
    text = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        obj = json.loads(
            text,
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=lambda _: (_ for _ in ()).throw(
                CandidateError("nonfinite JSON constant")
            ),
        )
    except (ValueError, TypeError, RecursionError) as exc:
        raise CandidateError("response is not one valid JSON object") from exc
    expected = {"code"} if mode == "code_only" else {"code", "strategy"}
    if (
        not isinstance(obj, dict)
        or set(obj) != expected
        or not isinstance(obj.get("code"), str)
    ):
        raise CandidateError(f"expected exactly fields {sorted(expected)}")
    strategy = obj.get("strategy")
    if mode == "strategy_code":
        if (
            not isinstance(strategy, str)
            or not strategy.strip()
            or len(strategy.split()) > 80
            or not strategy.isascii()
            or any(ord(c) < 32 and c not in "\n\r\t" for c in strategy)
            or not re.search(r"[A-Za-z]", strategy)
        ):
            raise CandidateError(
                "strategy must contain English ASCII text and at most 80 words"
            )
        strategy = " ".join(strategy.split())
    try:
        canonical = validate_code(obj["code"])
    except ValidationError as exc:
        raise CandidateError(f"code validation: {exc}") from exc
    return Candidate(canonical, strategy=strategy, raw_code=obj["code"])
