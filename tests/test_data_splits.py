# CS5491: Reasoning Augmented FunSearch
# Author: 李逸洋 (72655555)
# Responsibility: Benchmark preparation and evaluation scripts

import json
import shutil
from pathlib import Path

import pytest

from funsearch_project.datasets import load_split, manifest_hash, verify_data

ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_disjoint_frozen_sequences():
    seen = set()
    for split, count in [
        ("search", 100),
        ("validation", 50),
        ("test", 100),
        ("development", 4),
    ]:
        data = load_split(ROOT, split)
        assert len(data) == count
        for instance in data:
            assert len(instance.items) == 500
            assert instance.items not in seen
            seen.add(instance.items)
            assert instance.split == split
    assert verify_data(ROOT)


def test_or_originals_split_before_orders():
    seen = set()
    for split, count in [("search", 8), ("validation", 4), ("test", 8)]:
        data = load_split(ROOT, split, "or_library")
        sources = {i.source_id for i in data}
        assert len(sources) == count
        assert not sources & seen
        seen |= sources
        for source in sources:
            orders = [i for i in data if i.source_id == source]
            assert len(orders) == 3
            assert {i.order for i in orders} == {0, 1, 2}
            assert all(sorted(i.items) == sorted(orders[0].items) for i in orders)


def test_modified_frozen_data_rejected(tmp_path):
    shutil.copytree(ROOT / "data", tmp_path / "data")
    path = tmp_path / "data/processed/synthetic-search.json"
    rows = json.loads(path.read_text())
    rows[0]["items"][0] = 1
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="dataset hash mismatch"):
        load_split(tmp_path, "search")


def test_modified_manifest_rejected(tmp_path):
    shutil.copytree(ROOT / "data", tmp_path / "data")
    path = tmp_path / "data/manifests/manifest.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        manifest_hash(tmp_path)
