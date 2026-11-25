"""Milestone 3 / Lab 5.2 — persistent cosine search and metadata filtering.

Run from project root (the dataset & Lab 5.1 must already be present):
  python labs/lab_5_2_chroma_filters.py --index
  python labs/lab_5_2_chroma_filters.py --query "How do I secure an account?"
  python labs/lab_5_2_chroma_filters.py --query "How do I secure an account?" --category cybersecurity
  python labs/lab_5_2_chroma_filters.py --query "How do I secure an account?" --date-from 2026-01-01 --date-to 2026-12-31
  python labs/lab_5_2_chroma_filters.py --compare
  python labs/lab_5_2_chroma_filters.py --verify-persistence

The index is local and intentionally excluded from Git. Date metadata is stored
both as ISO text for display and as an integer YYYYMMDD for range comparisons.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from time import perf_counter

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from labs.lab_5_1_cosine_search import (
    DEFAULT_CACHE, DEFAULT_DATASET, MODEL_NAME, TEST_QUERIES,
    load_or_create_embeddings, load_records, search as numpy_search,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "chroma_db"
COLLECTION = "week5_documents_minilm_cosine"
REPORT = ROOT / "docs" / "lab_5_2_results.md"


def date_number(raw: str) -> int:
    """Validate strict ISO date and produce a numerically comparable YYYYMMDD."""
    if len(raw) != 10 or raw[4] != '-' or raw[7] != '-':
        raise ValueError(f"Expected date YYYY-MM-DD, received: {raw!r}")
    parsed = date.fromisoformat(raw)
    if parsed.isoformat() != raw:
        raise ValueError(f"Expected zero-padded ISO date, received: {raw!r}")
    return int(raw.replace('-', ''))


def metadata_filter(category: str | None = None, date_from: str | None = None,
                    date_to: str | None = None) -> dict | None:
    clauses = []
    if category:
        clauses.append({"category": category})
    first = date_number(date_from) if date_from else None
    last = date_number(date_to) if date_to else None
    if first is not None and last is not None and first > last:
        raise ValueError("--date-from must be on or before --date-to")
    if first is not None:
        clauses.append({"date_number": {"$gte": first}})
    if last is not None:
        clauses.append({"date_number": {"$lte": last}})
    if len(clauses) > 1:
        return {"$and": clauses}
    return clauses[0] if clauses else None


def record_metadata(record) -> dict:
    value = str(record.date)
    return {"category": str(record.category), "date": value,
            "date_number": date_number(value), "source": str(record.source)}


def corpus_signature(df) -> str:
    payload = json.dumps(list(zip(df["id"], df["text"])), ensure_ascii=False)
    return hashlib.sha256((MODEL_NAME + "\n" + payload).encode()).hexdigest()


def open_collection(db_dir: Path, reset: bool = False):
    client = chromadb.PersistentClient(path=str(db_dir))
    if reset:
        try:
            client.delete_collection(COLLECTION)
        except (ValueError, chromadb.errors.NotFoundError):
            pass
    collection = client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=None,  # always provide our own SentenceTransformer vectors
        configuration={"hnsw": {"space": "cosine"}},
    )
    # Chroma's cosine 'distance' is approximately 1 - cosine similarity.
    return collection


def index_corpus(collection, df, vectors: np.ndarray) -> int:
    if vectors.ndim != 2 or vectors.shape[0] != len(df):
        raise ValueError("One document vector must exist for every dataset row")
    existing = collection.get(include=[])["ids"]
    desired = set(df["id"].tolist())
    obsolete = [doc_id for doc_id in existing if doc_id not in desired]
    if obsolete:
        collection.delete(ids=obsolete)
    for start in range(0, len(df), 100):
        batch = df.iloc[start:start + 100]
        chunk = vectors[start:start + len(batch)]
        collection.upsert(
            ids=batch["id"].tolist(),
            documents=batch["text"].tolist(),
            embeddings=chunk.tolist(),
            metadatas=[record_metadata(r) for r in batch.itertuples(index=False)],
        )
    return collection.count()


def query_collection(collection, model, query: str, top_k: int = 5,
                     category: str | None = None, date_from: str | None = None,
                     date_to: str | None = None) -> tuple[list[dict], float]:
    if not query.strip():
        raise ValueError("Query cannot be blank")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    where = metadata_filter(category, date_from, date_to)
    count = collection.count()
    if count == 0:
        return [], 0.0
    start = perf_counter()
    embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    # Chroma applies the metadata filter before its nearest-neighbor ranking.
    response = collection.query(
        query_embeddings=[np.asarray(embedding, dtype=np.float32).tolist()],
        n_results=min(top_k, count), where=where,
        include=["documents", "metadatas", "distances"],
    )
    elapsed = perf_counter() - start
    hits = []
    for doc_id, doc, meta, distance in zip(
        response["ids"][0], response["documents"][0],
        response["metadatas"][0], response["distances"][0]
    ):
        hits.append({"id": doc_id, "text": doc, "category": meta["category"],
                     "date": meta["date"], "source": meta["source"],
                     "distance": float(distance), "similarity": 1.0 - float(distance)})
    return hits, elapsed


def print_hits(hits: list[dict], elapsed: float) -> None:
    print(f"\nResults ({len(hits)}) in {elapsed * 1000:.1f} ms")
    if not hits:
        print("No matches for these filters.")
    for idx, hit in enumerate(hits, 1):
        print(f"{idx}. {hit['id']} | similarity={hit['similarity']:.4f} | "
              f"category={hit['category']} | date={hit['date']}")
        print(f"   {hit['text']}")


def compare_searches(collection, model, df, vectors, output: Path = REPORT) -> None:
    lines = ["# Lab 5.2 — Chroma vs NumPy Search", "", "## Configuration", "",
             f"- Collection: `{COLLECTION}`", f"- Index path: `{DEFAULT_DB}`",
             f"- Model: `{MODEL_NAME}`", "- Distance: cosine (Chroma distance ≈ 1 − cosine similarity)",
             "- Input vectors: the exact normalized vectors produced/cached for Lab 5.1",
             "- Search: top 5, same queries; Chroma uses approximate nearest neighbors",
             "", "## Measured queries", "",
             "| Query | NumPy top-1 ID | Chroma top-1 ID | Top-5 overlap | NumPy ms | Chroma ms |",
             "|---|---|---|---:|---:|---:|"]
    for query, _expected in TEST_QUERIES:
        numpy_hits, numpy_elapsed = numpy_search(query, model, df, vectors, top_k=5)
        chroma_hits, chroma_elapsed = query_collection(collection, model, query, top_k=5)
        numpy_ids = numpy_hits["id"].tolist()
        chroma_ids = [hit["id"] for hit in chroma_hits]
        overlap = len(set(numpy_ids) & set(chroma_ids))
        lines.append(f"| {query.replace('|','/')} | {numpy_ids[0] if numpy_ids else '-'} | "
                     f"{chroma_ids[0] if chroma_ids else '-'} | {overlap}/5 | "
                     f"{numpy_elapsed * 1000:.1f} | {chroma_elapsed * 1000:.1f} |")
        print(f"{query}: overlap {overlap}/5")
    lines += ["", "## Filter checks (run separately and document observations)", "",
              "- Category: TODO — include query, category, and returned IDs.",
              "- Date: TODO — include range and returned dates.",
              "- Persistence: TODO — document separate-process index/reopen test.",
              "- Differences: Chroma's ANN index may reorder close vectors or differ slightly "
              "in floating-point scores from the NumPy exact scan.",
              "- Limitations: Demo corpus is synthetic. Document relevance needs manual review.", ""]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved comparison report to: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone 3: Chroma persistent cosine search")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB)
    parser.add_argument("--index", action="store_true", help="Create/update persisted collection")
    parser.add_argument("--rebuild-index", action="store_true", help="Delete and rebuild index")
    parser.add_argument("--query", type=str)
    parser.add_argument("--category", type=str)
    parser.add_argument("--date-from", type=str)
    parser.add_argument("--date-to", type=str)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--verify-persistence", action="store_true")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    if not any((args.index, args.rebuild_index, args.query is not None,
                args.verify_persistence, args.compare)):
        parser.error("Choose --index, --query, --verify-persistence, or --compare")
    try:
        where = metadata_filter(args.category, args.date_from, args.date_to)
        if args.top_k < 1:
            raise ValueError("--top-k must be >= 1")
        df = load_records(args.dataset)
        collection = open_collection(args.db_dir, reset=args.rebuild_index)
        if args.index or args.rebuild_index:
            model = SentenceTransformer(MODEL_NAME, device="cpu")
            vectors, seconds, from_cache = load_or_create_embeddings(model, df, DEFAULT_CACHE, False)
            current = index_corpus(collection, df, vectors)
            print(f"Indexed {current} records in {args.db_dir} ({'cached' if from_cache else 'new'} "
                  f"vectors, embed time {seconds:.2f}s).")
        if args.verify_persistence:
            reopened = open_collection(args.db_dir)
            print(f"Reopened collection: {reopened.count()} records")
            if reopened.count() != len(df):
                raise RuntimeError("Index row count differs from CSV. Run --index.")
        if args.query is not None or args.compare:
            if collection.count() != len(df):
                raise RuntimeError("Index missing/incomplete. Run --index first.")
            model = SentenceTransformer(MODEL_NAME, device="cpu")
            if args.query is not None:
                hits, elapsed = query_collection(
                    collection, model, args.query, args.top_k,
                    args.category, args.date_from, args.date_to)
                print(f"Query: {args.query!r}; filter: {where}")
                print_hits(hits, elapsed)
            if args.compare:
                vectors, _, _ = load_or_create_embeddings(model, df, DEFAULT_CACHE, False)
                compare_searches(collection, model, df, vectors)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
