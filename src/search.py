"""Query persisted embeddings with category/date filtering."""
from __future__ import annotations
from pathlib import Path
from src.data_loader import date_number
from src.embeddings import DEFAULT_MODEL, encode_texts
from src.indexing import DEFAULT_DB, model_collection_name, open_collection


def build_where(category: str | None = None, date_from: str | None = None,
                date_to: str | None = None) -> dict | None:
    start = date_number(date_from) if date_from else None
    end = date_number(date_to) if date_to else None
    if start is not None and end is not None and start > end:
        raise ValueError('Start date must be before or equal to end date')
    filters = []
    if category and category.strip():
        filters.append({'category': category.strip()})
    if start is not None:
        filters.append({'date_number': {'$gte': start}})
    if end is not None:
        filters.append({'date_number': {'$lte': end}})
    if len(filters) > 1:
        return {'$and': filters}
    return filters[0] if filters else None


def search_documents(query: str, top_k: int = 5, category: str | None = None,
                     date_from: str | None = None, date_to: str | None = None,
                     db_dir: Path = DEFAULT_DB, model_name: str = DEFAULT_MODEL,
                     chunk_words: int = 120, overlap_words: int = 20,
                     collection=None) -> list[dict]:
    if not query.strip():
        raise ValueError('Enter a non-empty search query')
    if not 1 <= int(top_k) <= 100:
        raise ValueError('Top-k must be between 1 and 100')
    where = build_where(category, date_from, date_to)
    if collection is None:
        name = model_collection_name(model_name, chunk_words, overlap_words)
        collection = open_collection(db_dir, name)
    if collection.count() == 0:
        return []
    vector = encode_texts([query.strip()], model_name)[0]
    output = collection.query(query_embeddings=[vector.tolist()], n_results=min(int(top_k), collection.count()),
                              where=where, include=['documents', 'metadatas', 'distances'])
    results = []
    for id_, text, meta, distance in zip(output['ids'][0], output['documents'][0],
                                         output['metadatas'][0], output['distances'][0]):
        results.append({'chunk_id': id_, 'document_id': meta.get('parent_id', id_),
                        'text': text, 'category': meta.get('category', ''),
                        'date': meta.get('date', ''), 'source': meta.get('source', ''),
                        'similarity': round(1.0 - float(distance), 4)})
    return results
