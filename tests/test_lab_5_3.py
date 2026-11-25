"""Fast pure-function tests — no pretrained model download needed."""
import numpy as np
import pandas as pd
import pytest
from labs.lab_5_3_clustering import (load_reviews, cluster_reviews,
    near_duplicate_pairs, classify_vectors)


def test_dataset_has_unique_ids_and_review_topics():
    df = load_reviews()
    assert len(df) == 126
    assert df['id'].is_unique
    assert df['category'].nunique() == 6


def test_cluster_outputs_and_representatives():
    x = np.array([[1, 0], [0.98, .02], [0, 1], [.02, .98]], dtype=float)
    labels, reps, _ = cluster_reviews(x, 2)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]
    assert len(reps) == 2


def test_invalid_cluster_count():
    with pytest.raises(ValueError):
        cluster_reviews(np.eye(3), 3)


def test_near_duplicates_not_self_and_sorted():
    x = np.array([[1, 0], [.98, .2], [0, 1]], dtype=float)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    pairs = near_duplicate_pairs(x, .90)
    assert len(pairs) == 1
    assert pairs[0][:2] == (0, 1)
    assert pairs[0][2] > .90


def test_threshold_validation():
    with pytest.raises(ValueError):
        near_duplicate_pairs(np.eye(3), 1.5)


def test_prototype_classifier_matches_nearest():
    x = np.array([[1, 0], [0, 1]], dtype=float)
    labels, scores = classify_vectors(x, np.eye(2), ['A', 'B'])
    assert labels == ['A', 'B']
    assert scores.shape == (2, 2)


def test_classifier_rejects_wrong_dimension():
    with pytest.raises(ValueError):
        classify_vectors(np.ones((1, 3)), np.ones((2, 4)), ['A', 'B'])
