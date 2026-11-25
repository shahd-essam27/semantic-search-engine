"""Quick tests for Milestone 3 plus optional real-Chroma persistence test."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from labs.lab_5_2_chroma_filters import (
    date_number, metadata_filter, index_corpus, open_collection,
    query_collection,
)


class FakeModel:
    def encode(self, texts, **kwargs):
        return np.array([[1., 0.]], dtype=np.float32)


def test_date_number_validation():
    assert date_number("2026-01-31") == 20260131
    for bad in ("2026-02-30", "2026-1-01", "2026-01-01x"):
        with pytest.raises(ValueError):
            date_number(bad)


def test_metadata_filter():
    assert metadata_filter("cybersecurity") == {"category": "cybersecurity"}
    assert metadata_filter(date_from="2026-01-01") == {"date_number": {"$gte": 20260101}}
    assert metadata_filter(date_to="2026-03-01") == {"date_number": {"$lte": 20260301}}
    assert metadata_filter("robotics", "2026-01-01", "2026-03-01") == {
        "$and": [{"category": "robotics"},
                 {"date_number": {"$gte": 20260101}},
                 {"date_number": {"$lte": 20260301}}]
    }
    with pytest.raises(ValueError):
        metadata_filter(date_from="2026-12-01", date_to="2026-01-01")


def test_persistence_filters_and_upsert(tmp_path: Path):
    db = tmp_path / "test_chroma"
    df = pd.DataFrame({
        "id": ["T1", "T2", "T3"], "text": ["secure a password", "cloud network", "protect an account"],
        "category": ["cybersecurity", "cloud", "cybersecurity"],
        "date": ["2026-01-01", "2026-02-01", "2026-03-01"],
        "source": ["unit_test"] * 3,
    })
    vectors = np.asarray([[1., 0.], [0., 1.], [.8, .6]], dtype=np.float32)
    col = open_collection(db)
    assert index_corpus(col, df, vectors) == 3
    assert index_corpus(col, df, vectors) == 3  # idempotent upsert
    col = open_collection(db)  # new client against same files
    assert col.count() == 3
    hits, _ = query_collection(col, FakeModel(), "security", category="cybersecurity")
    assert {x["id"] for x in hits} == {"T1", "T3"}
    hits, _ = query_collection(col, FakeModel(), "security", date_from="2026-02-01", date_to="2026-03-01")
    assert {x["id"] for x in hits} == {"T2", "T3"}
    hits, _ = query_collection(col, FakeModel(), "security", category="cloud", date_to="2026-01-31")
    assert hits == []
    assert hits == []
