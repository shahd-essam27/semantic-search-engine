"""Read and validate the project corpus."""
from __future__ import annotations
import csv
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / 'data' / 'raw' / 'documents.csv'
REQUIRED = ('id', 'text', 'category', 'date', 'source')


def date_number(value: str) -> int:
    if len(value) != 10 or date.fromisoformat(value).isoformat() != value:
        raise ValueError(f'Invalid date: {value!r}; expected YYYY-MM-DD')
    return int(value.replace('-', ''))


def load_documents(path: Path = DEFAULT_DATASET) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Dataset missing: {path}. Add a corpus CSV before indexing.')
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or any(k not in reader.fieldnames for k in REQUIRED):
            raise ValueError('Corpus must contain columns: ' + ', '.join(REQUIRED))
        rows = []
        ids = set()
        for num, row in enumerate(reader, start=2):
            record = {key: str(row[key] or '').strip() for key in REQUIRED}
            if not all(record.values()):
                raise ValueError(f'Missing required field at CSV line {num}')
            if record['id'] in ids:
                raise ValueError(f'Duplicate document ID: {record["id"]}')
            date_number(record['date'])
            ids.add(record['id'])
            rows.append(record)
    if not rows:
        raise ValueError('Dataset has no records')
    return rows


def doc_metadata(row: dict) -> dict:
    return {'category': row['category'], 'source': row['source'], 'date': row['date'], 'date_number': date_number(row['date']), 'parent_id': row['id']}


def split_chunks(text: str, chunk_words: int = 120, overlap_words: int = 20) -> list[str]:
    """Word-based chunking. No chunking for records <= chunk_words."""
    if not 0 <= overlap_words < chunk_words:
        raise ValueError('Require 0 <= overlap_words < chunk_words')
    words = text.split()
    if not words:
        raise ValueError('Cannot chunk blank text')
    if len(words) <= chunk_words:
        return [text]
    step = chunk_words - overlap_words
    return [' '.join(words[start:start + chunk_words])
            for start in range(0, len(words), step) if words[start:start + chunk_words]]


def prepare_chunks(rows: list[dict], chunk_words: int = 120, overlap_words: int = 20) -> list[dict]:
    prepared = []
    for row in rows:
        chunks = split_chunks(row['text'], chunk_words, overlap_words)
        for idx, text in enumerate(chunks):
            prepared.append({'id': f'{row["id"]}::chunk-{idx:03d}', 'text': text,
                             'metadata': {**doc_metadata(row), 'chunk_id': idx}})
    return prepared
