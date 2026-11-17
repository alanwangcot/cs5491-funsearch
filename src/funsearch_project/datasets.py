# CS5491: Reasoning Augmented FunSearch
# Author: 李逸洋 (72655555)
# Responsibility: Benchmark preparation and evaluation scripts

"""Immutable data preparation with explicit benchmark provenance and split integrity."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .binpacking import Instance

OR_URL = "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/binpack3.txt"
GENERATOR_VERSION = "new-weibull-rounded-rejection-v1"


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _download(url, path):
    if not path.exists():
        with urllib.request.urlopen(url, timeout=25) as response:
            path.write_bytes(response.read())
    return path.read_bytes()


def prepare_data(root: Path | str, config: dict | None = None, **kwargs):
    root = Path(root)
    data = root / "data"
    for part in ("raw", "processed", "manifests"):
        (data / part).mkdir(parents=True, exist_ok=True)
    manifest_path = data / "manifests" / "manifest.json"
    if manifest_path.exists():
        verify_data(root)
        return json.loads(manifest_path.read_text())
    config = config or {}
    settings = dict(config.get("dataset", config.get("data", {})))
    n_items = int(settings.get("n_items", 500))
    n_counts = {"search": 100, "validation": 50, "test": 100, "development": 4}
    seeds = {
        "search": 2026111601,
        "validation": 2026111602,
        "test": 2026111603,
        "development": 2026111604,
    }
    files = {}
    sequences = set()
    for split, count in n_counts.items():
        rng = np.random.default_rng(seeds[split])
        instances = []
        for index in range(count):
            while True:
                sizes = []
                while len(sizes) < n_items:
                    draws = np.rint(rng.weibull(3.0, n_items) * 45.0).astype(int)
                    sizes.extend(draws[(draws > 0) & (draws <= 100)].tolist())
                items = tuple(sizes[:n_items])
                h = _hash(json.dumps(items).encode())
                if h not in sequences:
                    sequences.add(h)
                    break
            instance = Instance(
                f"weibull-{split}-{index:03}", 100, items, split, benchmark="synthetic"
            )
            instances.append(asdict(instance))
        path = data / "processed" / f"synthetic-{split}.json"
        _write(path, instances)
        files[str(path.relative_to(root))] = {
            "sha256": _hash(path.read_bytes()),
            "count": count,
            "split": split,
            "benchmark": "synthetic",
            "seed": seeds[split],
        }
    raw_path = data / "raw" / "binpack3.txt"
    tokens = _download(OR_URL, raw_path).decode().split()
    count = int(tokens[0])
    offset = 1
    originals = []
    for _ in range(count):
        source_id = tokens[offset]
        capacity, n, best = map(int, tokens[offset + 1 : offset + 4])
        offset += 4
        items = tuple(map(int, tokens[offset : offset + n]))
        offset += n
        originals.append((source_id, capacity, items, best))
    if offset != len(tokens) or count != 20:
        raise ValueError("unexpected OR3 source format/count")
    permutation = np.random.default_rng(2026111690).permutation(count)
    memberships = {
        "search": permutation[:8].tolist(),
        "validation": permutation[8:12].tolist(),
        "test": permutation[12:].tolist(),
    }
    for split, indices in memberships.items():
        rows = []
        for index in indices:
            source_id, capacity, items, best = originals[index]
            for order in range(3):
                rng = np.random.default_rng(
                    np.random.SeedSequence([2026111691, index, order])
                )
                sequence = tuple((int(x) for x in rng.permutation(items)))
                rows.append(
                    asdict(
                        Instance(
                            f"{source_id}-order{order}",
                            capacity,
                            sequence,
                            split,
                            source_id,
                            order,
                            "or_library",
                        )
                    )
                )
        path = data / "processed" / f"or_library-{split}.json"
        _write(path, rows)
        files[str(path.relative_to(root))] = {
            "sha256": _hash(path.read_bytes()),
            "count": len(rows),
            "source_count": len(indices),
            "split": split,
            "benchmark": "or_library",
            "source_ids": [originals[i][0] for i in indices],
        }
    manifest = {
        "version": 1,
        "generator_version": GENERATOR_VERSION,
        "files": files,
        "synthetic": {
            "shape": 3.0,
            "scale": 45.0,
            "capacity": 100,
            "n_items": n_items,
            "rounding": "numpy.rint ties-to-even",
            "out_of_range": "reject <=0 or >100",
            "reference_deviation": "New synthetic benchmark; published notebook has literal n=5000 sequences but no generator parameters. This project uses explicit shape3/scale45 and n500.",
        },
        "or_library": {
            "url": OR_URL,
            "source_file_sha256": _hash(raw_path.read_bytes()),
            "orders_per_source": 3,
            "split_source_counts": {k: len(v) for k, v in memberships.items()},
        },
    }
    _write(manifest_path, manifest)
    (data / "manifests" / "manifest.sha256").write_text(
        _hash(manifest_path.read_bytes()) + "\n"
    )
    verify_data(root)
    return manifest


def manifest_hash(root: Path | str):
    path = Path(root) / "data" / "manifests" / "manifest.json"
    actual = _hash(path.read_bytes())
    expected = path.with_name("manifest.sha256").read_text().strip()
    if actual != expected:
        raise ValueError("immutable manifest hash mismatch")
    return actual


def verify_data(root: Path | str):
    root = Path(root)
    manifest_hash(root)
    manifest = json.loads((root / "data" / "manifests" / "manifest.json").read_text())
    for relative, spec in manifest["files"].items():
        if _hash((root / relative).read_bytes()) != spec["sha256"]:
            raise ValueError(f"immutable dataset hash mismatch: {relative}")
    return True


def load_split(
    root: Path | str, split: str, benchmark: str = "synthetic"
) -> list[Instance]:
    if benchmark == "orlibrary":
        benchmark = "or_library"
    root = Path(root)
    manifest_hash(root)
    path = root / "data" / "processed" / f"{benchmark}-{split}.json"
    manifest = json.loads((root / "data" / "manifests" / "manifest.json").read_text())
    expected = manifest["files"][str(path.relative_to(root))]["sha256"]
    if _hash(path.read_bytes()) != expected:
        raise ValueError(f"immutable dataset hash mismatch: {path.name}")
    return [Instance(**raw) for raw in json.loads(path.read_text())]
