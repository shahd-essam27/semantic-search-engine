"""Fast Milestone 5 tests; no model download or network needed."""
import csv
from pathlib import Path
import numpy as np
import pytest
from labs.lab_5_4_model_comparison import (
    normalize, topk, recall_at_k, hit_at_k, read_documents, read_queries,
    markdown_report,
)


def test_normalize_unit_vectors():
    v = normalize(np.array([[3., 4.], [0., 5.]]))
    assert np.allclose(np.linalg.norm(v,axis=1),1)


def test_top5_cosine_ranking():
    ids = ['a','b','c']
    vectors = normalize(np.array([[1,0],[0,1],[1,1]]))
    result, scores = topk(ids,vectors,normalize(np.array([[1,0]]))[0],2)
    assert result == ['a','c']
    assert scores[0] > scores[1]


def test_topk_rejects_mixed_spaces():
    with pytest.raises(ValueError, match='Dimension mismatch'):
        topk(['a'],np.array([[1.,2.,3.]]),np.array([1.,2.]))


def test_recall_vs_hit():
    assert recall_at_k(['a','c'],['a','b']) == 0.5
    assert hit_at_k(['a','c'],['a','b']) == 1.0
    assert hit_at_k(['x'],['a']) == 0.0


def test_data_files_have_valid_gold_ids():
    root = Path(__file__).resolve().parents[1]
    # Only test bundled evaluation set when the earlier milestone corpus is present.
    corpus = root / 'data/raw/documents.csv'
    evaluation = root / 'data/evaluation/model_comparison_queries.csv'
    if not corpus.exists():
        pytest.skip('Copy Milestone 1 documents.csv first')
    docs = read_documents(corpus)
    queries = read_queries(evaluation, {row['id'] for row in docs})
    assert len(queries)>=10
    assert {x['language'] for x in queries}=={'en','ar'}


def test_validation_rejects_bad_gold(tmp_path):
    path = tmp_path / 'q.csv'
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['query_id','language','query','relevant_ids','note'])
        w.writeheader();w.writerow(dict(query_id='x',language='ar',query='سؤال',relevant_ids='missing',note=''))
    with pytest.raises(ValueError,match='Unknown relevant IDs'):
        read_queries(path, {'exists'})


def test_reports_keep_user_decision_unfilled():
    s = dict(model_name='x',dimension=384,model_weights_mb='N/A',device='cpu',
             model_load_s=1.,corpus_embed_s=2.,query_avg_ms=3.,rss_before_mb=None,
             rss_after_mb=None,english_recall=.5,arabic_recall=.2,english_hit=1.,
             arabic_hit=.5,overall_recall=.35)
    line = dict(model='english',query_id='en-1',language='en',recall_at_5=.5,top5_ids='a;b')
    report = markdown_report([s,s], [line,{**line,'model':'multilingual','language':'ar'}],2,[line])
    assert 'TODO' in report and 'English vs Multilingual' in report
