# CS5491: Reasoning Augmented FunSearch
# Author: 李逸洋 (72655555)
# Responsibility: Benchmark preparation and evaluation scripts

"""Fail-closed, credential-free Docker execution of generated programs."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

from .baselines import best_fit
from .binpacking import Instance, pack_instance

IMAGE = "funsearch-evaluator:1"


@dataclass
class EvaluationResult:
    status: str
    fitness: float | None = None
    bins: dict[str, int] = field(default_factory=dict)
    ratios: dict[str, float] = field(default_factory=dict)
    duration_seconds: float = 0
    error: str | None = None
    instance_executions: int = 0
    smoke_executions: int = 0

    def to_dict(self):
        return asdict(self)


def isolation_available(image: str = IMAGE) -> bool:
    try:
        return (
            subprocess.run(
                ["docker", "image", "inspect", image], capture_output=True, timeout=10
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


def build_image(root: Path | str):
    subprocess.run(
        [
            "docker",
            "build",
            "-f",
            str(Path(root) / "Dockerfile.evaluator"),
            "-t",
            IMAGE,
            str(Path(root)),
        ],
        check=True,
    )


@lru_cache(maxsize=12)
def _best_fit_cached(serialized: str):
    instances = [Instance(**x) for x in json.loads(serialized)]
    return {
        x.instance_id: pack_instance(best_fit, x, diagnostics=False).bins_used
        for x in instances
    }


def dataset_hash(dataset) -> str:
    return hashlib.sha256(
        json.dumps([asdict(x) for x in dataset], sort_keys=True).encode()
    ).hexdigest()


def evaluate_candidate(
    code: str, dataset: list[Instance], limits: dict | None = None
) -> EvaluationResult:
    from .validation import ValidationError, validate_code

    started = time.monotonic()
    limits = dict(limits or {})
    try:
        code = validate_code(code)
    except (ValidationError, SyntaxError, ValueError) as exc:
        return EvaluationResult(
            "static_error", error=str(exc), duration_seconds=time.monotonic() - started
        )
    if not dataset or any(not x.items for x in dataset):
        return EvaluationResult(
            "dataset_error", error="metric requires nonempty instances and dataset"
        )
    if len({x.instance_id for x in dataset}) != len(dataset):
        return EvaluationResult("dataset_error", error="duplicate instance identifiers")
    image = limits.get("image", IMAGE)
    if not isolation_available(image):
        return EvaluationResult(
            "isolation_unavailable",
            error=f"Docker image {image} unavailable; no host fallback",
        )
    serialized = json.dumps([asdict(x) for x in dataset], sort_keys=True)
    reference = _best_fit_cached(serialized)
    payload = json.dumps(
        {
            "code": code,
            "dataset": json.loads(serialized),
            "limits": limits,
            "best_fit_bins": reference,
        }
    )
    name = "funsearch-" + uuid.uuid4().hex
    source = Path(__file__).resolve().parent
    command = [
        "docker",
        "run",
        "--rm",
        "--interactive",
        "--name",
        name,
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "32",
        "--memory",
        str(limits.get("memory", "512m")),
        "--memory-swap",
        str(limits.get("memory", "512m")),
        "--cpus",
        "1",
        "--user",
        "65534:65534",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=16m",
        "--mount",
        f"type=bind,source={source},target=/app/funsearch_project,readonly",
        image,
    ]
    try:
        result = subprocess.run(
            command,
            input=payload,
            text=True,
            capture_output=True,
            timeout=float(limits.get("total_seconds", 120)),
        )
        if result.returncode:
            return EvaluationResult(
                "worker_crash",
                error=result.stderr[-1000:],
                duration_seconds=time.monotonic() - started,
            )
        parsed = json.loads(result.stdout)
        parsed["duration_seconds"] = time.monotonic() - started
        return EvaluationResult(**parsed)
    except subprocess.TimeoutExpired:
        return EvaluationResult(
            "timeout",
            error="total evaluation wall time exceeded",
            duration_seconds=time.monotonic() - started,
        )
    except (OSError, ValueError, TypeError) as exc:
        return EvaluationResult(
            "worker_error", error=str(exc), duration_seconds=time.monotonic() - started
        )
    finally:
        # CLI timeout can leave a container alive; kill by unique generated name.
        subprocess.run(
            ["docker", "rm", "--force", name], capture_output=True, timeout=10
        )
