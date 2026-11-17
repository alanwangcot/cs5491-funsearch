# CS5491: Reasoning Augmented FunSearch
# Author: 朱泳颐 (72657023)
# Responsibility: Search framework and logging

"""Public experiment configuration and immutable protocol fingerprints."""

import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path

import yaml

from .storage import atomic_json, digest, utcnow

DEFAULTS = {
    "experiment_id": "main",
    "attempts_per_run": 200,
    "global_cap": 2200,
    "seeds": [11, 22, 33, 44, 55],
    "modes": ["code_only", "strategy_code"],
    "population_size": 20,
    "tournament_size": 3,
    "workers": 4,
    "provider": {
        "name": "deepseek",
        "model": "deepseek-flash",
        "base_url": "https://api.deepseek.com",
        "env_path": "../project/.env",
        "temperature": 1.0,
        "max_tokens": 2048,
        "timeout": 90,
        "thinking": "disabled",
    },
    "limits": {"total_seconds": 600, "memory": "512m", "instance_seconds": 5},
}


def load_config(config):
    if isinstance(config, dict):
        result = deepcopy(config)
        result.setdefault("root", str(Path.cwd()))
        return result
    path = Path(config).resolve()
    result = deepcopy(DEFAULTS)
    supplied = yaml.safe_load(path.read_text()) or {}
    for key, value in supplied.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key].update(value)
        else:
            result[key] = value
    result["root"] = str(path.parent.parent)
    result["config_path"] = str(path)
    return result


def public_config(config):
    return {k: v for k, v in config.items() if k not in {"root", "config_path"}}


def protocol_fingerprint(config):
    root = Path(config["root"])
    files = (
        list((root / "prompts").glob("*.txt"))
        + list((root / "src/funsearch_project").glob("*.py"))
        + list((root / "data/manifests").glob("*.json"))
    )
    files += [
        root / name
        for name in (
            "pyproject.toml",
            "uv.lock",
            ".python-version",
            "Dockerfile.evaluator",
        )
        if (root / name).is_file()
    ]
    hashes = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(files)
    }
    runtime = {}
    if config.get("provider", {}).get("name") != "fake":
        image = config.get("limits", {}).get("image", "funsearch-evaluator:1")
        inspected = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if inspected.returncode:
            raise RuntimeError("Cannot fingerprint evaluator Docker image")
        runtime["execution_image_id"] = inspected.stdout.strip()
    return {
        "config": public_config(config),
        "files": hashes,
        "runtime": runtime,
        "hash": digest([public_config(config), hashes, runtime]),
    }


def freeze_protocol(config):
    root = Path(config["root"])
    snapshot = protocol_fingerprint(config)
    snapshot["frozen_at"] = utcnow()
    atomic_json(root / "runs/protocol.json", snapshot)
    return snapshot


def require_frozen(config):
    path = Path(config["root"]) / "runs/protocol.json"
    if not path.exists():
        raise RuntimeError("Production requires freeze before any API calls")
    frozen = json.loads(path.read_text())
    if frozen["hash"] != protocol_fingerprint(config)["hash"]:
        raise RuntimeError("Protocol/config/code/data/prompts changed after freeze")
    return frozen


def verify_runtime(config):
    """Check that the evaluator can run a small packing example."""
    from .baselines import BEST_FIT_CODE
    from .binpacking import Instance
    from .evaluator import evaluate_candidate

    result = evaluate_candidate(
        BEST_FIT_CODE,
        [Instance("preflight", 10, (6, 6, 4, 4))],
        config.get("limits", {}),
    )
    if result.status != "ok" or result.bins.get("preflight") != 2:
        raise RuntimeError(f"Evaluator setup failed: {result}")
    return config.get("provider", {})
