"""Fast tests for Milestone 2: no model download or CPU-intensive embedding needed."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from labs.lab_5_1_cosine_search import load_records, search


class TinyFakeModel:
    def encode(self, texts, **_kwargs):
        # The provided document vectors are already normalized.
        return np.array([[1.0, 0.0]], dtype=np.float32)


def test_search_ranks_cosine_scores():
    docs = pd.DataFrame({
        "id": ["D1", "D2", "D3"], "text": ["a", "b", "c"],
        "category": ["a", "b", "c"]
    })
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.6]], dtype=np.float32)
    hits, elapsed = search("test", TinyFakeModel(), docs, vectors, top_k=2)
    assert hits["id"].tolist() == ["D1", "D3"]
    assert len(hits) == 2
    assert elapsed >= 0


def test_rejects_blank_query():
    with pytest.raises(ValueError, match="Query must not be empty"):
        search(" ", TinyFakeModel(), pd.DataFrame(), np.zeros((0, 2), dtype=np.float32))


def test_load_missing_dataset(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_records(tmp_path / "does_not_exist.csv")
