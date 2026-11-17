# CS5491: Reasoning Augmented FunSearch
# Author: 王宇浩 (72655080)
# Responsibility: LLM interface and strategy augmentation

import json

import numpy as np
import pytest

from funsearch_project.candidates import CandidateError, parse_candidate
from funsearch_project.prompts import build_prompt
from funsearch_project.validation import (
    ValidationError,
    compile_candidate,
    validate_code,
)

BF = "def score_bins(item, remaining):\n    return -(remaining - item)\n"


def test_canonicalization_removes_description():
    decorated = 'def score_bins(item, remaining):\n    """Favor smallest gap."""\n    # This text must not reach code-only descendants.\n    return -(remaining-item)\n'
    assert validate_code(decorated) == validate_code(BF)
    fn = compile_candidate(decorated)
    np.testing.assert_array_equal(fn(2, np.array([4, 5])), [-2, -3])


@pytest.mark.parametrize(
    "code",
    [
        "import os\n" + BF,
        BF + "\nx = 5\n",
        'def score_bins(item, remaining):\n    return open("x")\n',
        'def score_bins(item, remaining):\n    return np.load("x")\n',
        "def score_bins(item, remaining):\n    return remaining.__class__\n",
        "def score_bins(item, remaining):\n    while True:\n        pass\n    return remaining\n",
        "def score_bins(item, remaining):\n    return [x for x in remaining]\n",
        "def score_bins(item, remaining):\n    remaining[0] = 0\n    return remaining\n",
        "def score_bins(item, remaining):\n    remaining += 1\n    return remaining\n",
        "def score_bins(item, remaining):\n    return np.add(remaining, 1, out=remaining)\n",
        "def score_bins(item, remaining):\n    return score_bins(item, remaining)\n",
        "def score_bins(item, remaining):\n    def helper():\n        return 1\n    return remaining\n",
        "@np.mean\ndef score_bins(item, remaining):\n    return remaining\n",
        "def score_bins(item, remaining=1):\n    return remaining\n",
        "def score_bins(item, remaining):\n    global x\n    return remaining\n",
        "def score_bins(item, remaining):\n    return np.random.random(remaining.shape)\n",
        "def score_bins(item, remaining):\n    f = np.mean\n    return f(remaining)\n",
    ],
)
def test_reject_forbidden_programs(code):
    with pytest.raises(ValidationError):
        validate_code(code)


def test_vector_numeric_program_allowed():
    code = """def score_bins(item, remaining):
    residual = remaining - item
    mean = np.mean(residual)
    bonus = np.where(residual == 0, 1000.0, 0.0)
    return bonus - residual / (mean + 1.0) + 0.1 * np.sin(residual)
"""
    assert np.isfinite(compile_candidate(code)(2, np.array([2, 4, 6]))).all()


def test_parse_one_json_and_strategy_identity():
    a = parse_candidate(
        json.dumps({"code": BF, "strategy": "Prefer the smallest residual."}),
        "strategy_code",
    )
    b = parse_candidate(
        json.dumps({"code": BF, "strategy": "Prioritize exact fills."}), "strategy_code"
    )
    assert a.code_hash == b.code_hash
    assert a.identity != b.identity
    fenced = "```json\n" + json.dumps({"code": BF}) + "\n```"
    assert parse_candidate(fenced, "code_only").code_hash == a.code_hash


@pytest.mark.parametrize(
    "text, mode",
    [
        ('[{"code":"x"}]', "code_only"),
        ('{"code":"x"}{"code":"x"}', "code_only"),
        ('{"code":"x","code":"y"}', "code_only"),
        (json.dumps({"code": BF, "strategy": "unexpected"}), "code_only"),
        (json.dumps({"code": BF, "strategy": "word " * 81}), "strategy_code"),
        (json.dumps({"code": BF, "strategy": "贪心策略"}), "strategy_code"),
        (json.dumps({"code": BF, "strategy": ""}), "strategy_code"),
        (json.dumps({"code": BF}), "strategy_code"),
    ],
)
def test_reject_bad_envelopes(text, mode):
    with pytest.raises(CandidateError):
        parse_candidate(text, mode)


def test_treatment_boundary_has_no_inherited_strategy_in_control():
    parent = parse_candidate(
        json.dumps({"code": BF, "strategy": "Secret description sentinel."}),
        "strategy_code",
    )
    control = build_prompt(parent, "code_only", 1.0)
    augmented = build_prompt(parent, "strategy_code", 1.0)
    assert control[0] == augmented[0]
    assert "Secret description sentinel" not in str(control)
    assert "Secret description sentinel" in str(augmented)
    assert parent.code in control[1]["content"]
    assert parent.code in augmented[1]["content"]
