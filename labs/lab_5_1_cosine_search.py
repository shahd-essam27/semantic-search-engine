"""Week 5 / Milestone 2 / Lab 5.1: local cosine-similarity semantic search.

Run from the repository root:
  python labs/lab_5_1_cosine_search.py --query "How can I secure my accounts?"
  python labs/lab_5_1_cosine_search.py --evaluate
  python labs/lab_5_1_cosine_search.py  # interactive mode

No vector database is used here. All embeddings use one consistent model.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "raw" / "documents.csv"
DEFAULT_CACHE = ROOT / "data" / "processed" / "lab_5_1"
DEFAULT_RESULTS = ROOT / "docs" / "lab_5_1_results.md"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
REQUIRED_COLUMNS = {"id", "text", "category", "date", "source"}

# Intentionally includes paraphrases with limited exact keyword overlap.
TEST_QUERIES = [
    ("How do I protect my online accounts?", "cybersecurity"),
    ("Why can't two devices communicate across different networks?", "networking"),
    ("How can I prevent Python errors from stopping my program?", "python"),
    ("How do algorithms group similar observations without labels?", "machine_learning"),
    ("What does it mean when a language model makes up facts?", "llms"),
    ("How do I find documents that mean the same thing as my question?", "semantic_search"),
    ("How are records organized into tables and fields?", "databases"),
    ("How can I deploy my app without managing physical servers?", "cloud"),
    ("How does a machine know where it is while moving?", "robotics"),
    ("How can wearable sensors monitor health?", "health_technology"),
]


def load_records(path: Path) -> pd.DataFrame:
    """Load and validate the same 200+ record corpus from Milestone 1."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Copy the Milestone 1 data/raw/documents.csv file to this location."
        )
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {', '.join(sorted(missing))}")
    if len(df) < 200:
        raise ValueError(f"At least 200 records required; found {len(df)}")
    if df["id"].str.strip().eq("").any() or df["text"].str.strip().eq("").any():
        raise ValueError("Document IDs and texts must not be empty")
    if df["id"].duplicated().any():
        raise ValueError("Document IDs must be unique")
    return df.reset_index(drop=True)


def load_or_create_embeddings(model: SentenceTransformer, df: pd.DataFrame,
                              cache_dir: Path, rebuild: bool) -> tuple[np.ndarray, float, bool]:
    """Compute normalized embeddings; cache only when the entire corpus signature matches."""
    import hashlib
    import json

    # Includes IDs and texts to invalidate if content or row order changes.
    payload = json.dumps(list(zip(df["id"], df["text"])), ensure_ascii=False).encode("utf-8")
    signature = hashlib.sha256(MODEL_NAME.encode() + b"\n" + payload).hexdigest()
    cache_file = cache_dir / "embeddings.npy"
    meta_file = cache_dir / "metadata.json"

    if not rebuild and cache_file.exists() and meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if meta.get("signature") == signature:
            vectors = np.load(cache_file, allow_pickle=False)
            if vectors.shape[0] == len(df) and vectors.ndim == 2:
                return vectors, 0.0, True

    start = perf_counter()
    vectors = model.encode(
        df["text"].tolist(),
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    elapsed = perf_counter() - start
    vectors = np.asarray(vectors, dtype=np.float32)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, vectors)
    meta_file.write_text(json.dumps({"signature": signature, "model": MODEL_NAME,
                                     "records": len(df), "dimension": vectors.shape[1]}, indent=2),
                         encoding="utf-8")
    return vectors, elapsed, False


def search(query: str, model: SentenceTransformer, df: pd.DataFrame,
           doc_vectors: np.ndarray, top_k: int = 5) -> tuple[pd.DataFrame, float]:
    """Compare the query against the entire corpus with normalized cosine similarity."""
    if not query.strip():
        raise ValueError("Query must not be empty")
    start = perf_counter()
    query_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    # Dot product equals cosine similarity for L2-normalized vectors.
    scores = doc_vectors @ query_vec
    order = np.argsort(-scores, kind="stable")[:min(top_k, len(df))]
    results = df.iloc[order][["id", "text", "category"]].copy()
    results.insert(0, "rank", np.arange(1, len(results) + 1))
    results.insert(2, "score", scores[order])
    return results, perf_counter() - start


def print_hits(query: str, hits: pd.DataFrame, seconds: float) -> None:
    print(f"\nQuery: {query!r} | Search time: {seconds * 1000:.1f} ms")
    for row in hits.itertuples(index=False):
        print(f"  {row.rank}. [{row.id}] score={row.score:.4f} category={row.category}")
        print(f"     {row.text}")


def escape_md(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").replace("`", "'")


def run_evaluation(model: SentenceTransformer, df: pd.DataFrame,
                   vectors: np.ndarray, model_load_s: float, embed_s: float,
                   cache_hit: bool, output: Path) -> None:
    lines = [
        "# Lab 5.1 — Cosine Search Results",
        "",
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "## Configuration",
        f"- Model: `{MODEL_NAME}`",
        f"- Device: `{model.device}`",
        f"- Record count: {len(df)}",
        f"- Embedding shape: `{tuple(vectors.shape)}`",
        f"- Embedding dtype: `{vectors.dtype}`",
        f"- Model loading time: {model_load_s:.3f} s (local run; can vary)",
        f"- Corpus embedding time: {embed_s:.3f} s" + (" (cached embeddings reused)" if cache_hit else ""),
        "- Search approach: normalized embeddings + NumPy dot product (cosine similarity)",
        "- Search top-k: 5",
        "", "## Ten test queries", "",
        "| # | Query | Expected topic | Top result category | Top result score | Latency (ms) |",
        "|---|---|---|---|---:|---:|",
    ]
    top_correct = 0
    latencies = []
    for idx, (query, expected_category) in enumerate(TEST_QUERIES, 1):
        hits, elapsed = search(query, model, df, vectors)
        first = hits.iloc[0]
        actual = str(first["category"])
        top_correct += actual == expected_category
        latencies.append(elapsed * 1000)
        lines.append(f"| {idx} | {escape_md(query)} | {expected_category} | {actual} | "
                     f"{float(first['score']):.4f} | {elapsed*1000:.1f} |")
        print_hits(query, hits, elapsed)
    lines += [
        "", "## Basic diagnostic (not a gold-standard relevance metric)",
        f"- Top-1 category matched a suggested topic for {top_correct}/{len(TEST_QUERIES)} queries.",
        f"- Average observed query latency: {float(np.mean(latencies)):.1f} ms.",
        "- Category agreement **does not establish document-level relevance**. Manually inspect the top five.",
        "", "## Manual review and limitations (complete after reading the results)",
        "- **Relevant results:** TODO — cite 2 query/result examples.",
        "- **Weak results:** TODO — cite 2 irrelevant or surprising results.",
        "- **Paraphrases:** TODO — assess whether meaning was retrieved without exact keywords.",
        "- **Dataset limitation:** The Milestone 1 dataset is synthetic demonstration notes. "
        "Replace with a real, properly licensed corpus before claiming the real-corpus requirement is complete.",
        "- **Timing limitation:** First runs are affected by model loading, hardware, and caching.",
        "- **Next milestone:** Use the same corpus and model in Chroma; never mix vectors from different models.",
        "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved evaluation report to: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone 2: semantic search using cosine similarity")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--query", type=str, help="Search once and exit")
    parser.add_argument("--evaluate", action="store_true", help="Test 10 queries and write results markdown")
    parser.add_argument("--rebuild", action="store_true", help="Regenerate and recache the corpus embeddings")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    try:
        df = load_records(args.dataset)
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    print(f"Loaded {len(df)} documents from {args.dataset}")

    start = perf_counter()
    model = SentenceTransformer(MODEL_NAME, device=args.device)
    model_load_s = perf_counter() - start
    vectors, embed_s, hit = load_or_create_embeddings(model, df, args.cache_dir, args.rebuild)
    print(f"Model: {MODEL_NAME} | Device: {model.device}")
    print(f"Embeddings: shape={vectors.shape}, dtype={vectors.dtype}")
    print(f"Model load: {model_load_s:.2f}s | corpus encoding: {embed_s:.2f}s"
          + (" (cached)" if hit else " (new)"))

    if args.evaluate:
        run_evaluation(model, df, vectors, model_load_s, embed_s, hit, DEFAULT_RESULTS)
    elif args.query is not None:
        try:
            hits, elapsed = search(args.query, model, df, vectors)
        except ValueError as exc:
            parser.exit(2, f"ERROR: {exc}\n")
        print_hits(args.query, hits, elapsed)
    else:
        print("\nInteractive search. Type a question, or 'exit' to stop.")
        while True:
            try:
                query = input("\nSearch> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break
            if query.lower() in {"exit", "quit"}:
                break
            if not query:
                print("Please enter a non-empty search query.")
                continue
            hits, elapsed = search(query, model, df, vectors)
            print_hits(query, hits, elapsed)


if __name__ == "__main__":
    main()
