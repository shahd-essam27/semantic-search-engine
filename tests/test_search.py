"""Fast isolated tests. Full model and Chroma tests must also be run locally."""
import csv
from pathlib import Path
import pytest
from src.data_loader import date_number, load_documents, prepare_chunks, split_chunks
from src.indexing import model_collection_name
from src.search import build_where, search_documents
from app import ui_search


def test_date_numeric():
    assert date_number('2026-01-09') == 20260109
    with pytest.raises(ValueError):
        date_number('2026-02-30')


def test_chunking_short_document():
    assert split_chunks('one two three', 10, 2) == ['one two three']


def test_chunk_ids_and_metadata():
    rows = [{'id': 'D1', 'text': 'a b c d e f', 'category': 'x', 'date': '2026-01-01', 'source': 'demo'}]
    result = prepare_chunks(rows, chunk_words=4, overlap_words=1)
    assert result[0]['id'] == 'D1::chunk-000'
    assert result[1]['metadata']['parent_id'] == 'D1'
    assert result[0]['metadata']['date_number'] == 20260101


def test_bad_overlap():
    with pytest.raises(ValueError):
        split_chunks('text', 5, 5)


def test_metadata_filters():
    assert build_where('networking') == {'category': 'networking'}
    assert build_where('networking', '2026-01-01', '2026-03-01') == {'$and': [
        {'category': 'networking'}, {'date_number': {'$gte': 20260101}},
        {'date_number': {'$lte': 20260301}}]}
    with pytest.raises(ValueError):
        build_where(date_from='2026-04-01', date_to='2026-01-01')


def test_separate_model_collection_ids():
    assert model_collection_name('model_a', 120, 20) != model_collection_name('model_b', 120, 20)
    assert model_collection_name('model_a', 120, 20) != model_collection_name('model_a', 200, 20)


def test_dataset_duplicate_rejected(tmp_path):
    path = tmp_path / 'docs.csv'
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'text', 'category', 'date', 'source'])
        for _ in range(2):
            writer.writerow(['D1', 'hello', 'x', '2026-01-01', 'test'])
    with pytest.raises(ValueError, match='Duplicate'):
        load_documents(path)


def test_empty_index_returns_empty_without_loading_model():
    class EmptyCollection:
        def count(self): return 0
    assert search_documents('hello', collection=EmptyCollection()) == []


def test_blank_query_rejected():
    assert ui_search('   ', 5, '', '', '') == ('Please type a search query.', [])
    with pytest.raises(ValueError):
        search_documents(' ', collection=object())


def test_invalid_filter_ui():
    message, table = ui_search('hello', 5, '', '2026-99-99', '')
    assert 'Invalid input' in message
    assert table == []


def test_relevance_scoring():
    from scripts.evaluate_search import score_retrieval
    assert score_retrieval({'A', 'B'}, ['A', 'X', 'Y']) == {'hit': 1, 'recall': 0.5}
    assert score_retrieval({'A'}, []) == {'hit': 0, 'recall': 0.0}
