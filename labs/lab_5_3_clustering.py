r"""Week 5 / Milestone 4 / Lab 5.3: clustering plus two embedding applications.

Run from project root:
    .\.venv\Scripts\python.exe -m labs.lab_5_3_clustering --all
    .\.venv\Scripts\python.exe -m labs.lab_5_3_clustering --cluster --k 6
    .\.venv\Scripts\python.exe -m labs.lab_5_3_clustering --dedupe --threshold 0.90
    .\.venv\Scripts\python.exe -m labs.lab_5_3_clustering --classify --text "Shipment arrived late"

The dataset is SYNTHETIC DEMONSTRATION DATA, not genuine customer feedback.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "raw" / "customer_reviews.csv"
DEFAULT_REPORT = ROOT / "docs" / "lab_5_3_results.md"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CLASS_DESCRIPTIONS = {
    "battery": "A customer review about battery life, charging, or electrical power consumption.",
    "delivery": "A customer review about parcel shipping, couriers, tracking, or delivery time.",
    "customer_support": "A customer review about contacting support agents, help desks, warranties, or refunds.",
    "usability": "A customer review about the user interface, navigation, setup, or ease of use.",
    "value": "A customer review about price, money, subscriptions, discounts, or affordability.",
    "build_quality": "A customer review about the product materials, physical durability, or construction.",
}


def load_reviews(path: Path = DEFAULT_DATA) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing reviews file: {path}. Copy the Milestone 4 ZIP into the project root.")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"id", "text", "category", "source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    if len(df) < 20:
        raise ValueError("Need at least 20 reviews for the clustering demonstration")
    if df["id"].duplicated().any() or df["id"].str.strip().eq("").any():
        raise ValueError("Review IDs must be nonempty and unique")
    if df["text"].str.strip().eq("").any():
        raise ValueError("Review text cannot be empty")
    return df.reset_index(drop=True)


def normalized_embeddings(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """L2-normalized vectors: dot product is cosine similarity."""
    if not texts or any(not text.strip() for text in texts):
        raise ValueError("Provide at least one nonempty text")
    values = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                          batch_size=32, show_progress_bar=False)
    return np.asarray(values, dtype=np.float32)


def cluster_reviews(vectors: np.ndarray, n_clusters: int = 6, random_state: int = 42):
    """Fit KMeans; return assignments, representative positions, and silhouette if possible."""
    if vectors.ndim != 2 or not 2 <= n_clusters < len(vectors):
        raise ValueError("k must be >=2 and smaller than number of reviews")
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(vectors)
    reps = {}
    for cluster_id in range(n_clusters):
        members = np.flatnonzero(labels == cluster_id)
        distances = np.linalg.norm(vectors[members] - km.cluster_centers_[cluster_id], axis=1)
        reps[cluster_id] = members[np.argsort(distances)[:3]].tolist()
    if 1 < len(set(labels.tolist())) < len(vectors):
        silhouette = float(silhouette_score(vectors, labels, metric="cosine"))
    else:
        silhouette = None
    return labels, reps, silhouette


def near_duplicate_pairs(vectors: np.ndarray, threshold: float = 0.90) -> list[tuple[int, int, float]]:
    """All pairs over threshold. O(n^2): appropriate for this SMALL lab corpus only."""
    if vectors.ndim != 2 or not 0 <= threshold <= 1:
        raise ValueError("Expected 2D vectors and a threshold between 0 and 1")
    similarities = vectors @ vectors.T
    row_idx, col_idx = np.triu_indices(len(vectors), k=1)
    mask = similarities[row_idx, col_idx] >= threshold
    results = [(int(i), int(j), float(similarities[i, j]))
               for i, j in zip(row_idx[mask], col_idx[mask])]
    return sorted(results, key=lambda x: -x[2])


def classify_vectors(review_vectors: np.ndarray, label_vectors: np.ndarray,
                     labels: list[str]) -> tuple[list[str], np.ndarray]:
    """Prototype similarity classification, NOT a trained NLI zero-shot classifier."""
    if review_vectors.ndim != 2 or label_vectors.ndim != 2:
        raise ValueError("Both inputs must be 2-dimensional")
    if review_vectors.shape[1] != label_vectors.shape[1] or len(labels) != len(label_vectors):
        raise ValueError("Incompatible embedding dimensions or label count")
    score = review_vectors @ label_vectors.T
    return [labels[i] for i in np.argmax(score, axis=1)], score


def safe(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ").replace("`", "'")


def build_report(df: pd.DataFrame, vectors: np.ndarray, model: SentenceTransformer,
                 k: int, threshold: float, example_text: str, seconds: float) -> str:
    labels, representatives, silhouette = cluster_reviews(vectors, k)
    duplicates = near_duplicate_pairs(vectors, threshold)
    class_names = list(CLASS_DESCRIPTIONS)
    prototype_vectors = normalized_embeddings(model, list(CLASS_DESCRIPTIONS.values()))
    predicted, scores = classify_vectors(vectors, prototype_vectors, class_names)
    sample_pred, sample_scores = classify_vectors(
        normalized_embeddings(model, [example_text]), prototype_vectors, class_names)
    accuracy = sum(p == y for p, y in zip(predicted, df["category"])) / len(df)
    lines = [
        "# Milestone 4 — Lab 5.3 Results", "",
        f"Generated locally: {datetime.now().astimezone().isoformat(timespec='seconds')}", "",
        "## Setup", "",
        f"- Dataset: `{DEFAULT_DATA.relative_to(ROOT)}` ({len(df)} original synthetic demonstration reviews)",
        f"- Embedding model: `{MODEL_NAME}` (CPU)",
        f"- Embedding dimension: {vectors.shape[1]}",
        f"- Measured embedding duration: {seconds:.3f} seconds (device-dependent)",
        f"- KMeans clusters (`k`): {k}; `random_state=42`, `n_init=10`",
        f"- Cosine near-duplicate threshold: {threshold:.2f}",
        "", "## KMeans clusters — interpret examples manually", "",
        "**Important:** KMeans is unsupervised; numeric cluster IDs have no inherent meaning. "
        "`category` below comes from synthetic dataset labels and is shown only as a diagnostic hint. "
        "Read the representatives and write your own human-readable interpretation.", "",
    ]
    for cluster in range(k):
        indices = np.flatnonzero(labels == cluster)
        diagnostic = df.iloc[indices]["category"].value_counts().head(3)
        diagnostics = ", ".join(f"{cat} ({count})" for cat, count in diagnostic.items())
        lines.extend([f"### Cluster {cluster} ({len(indices)} reviews)",
                      f"- Existing category breakdown (diagnostic only): {diagnostics}",
                      "- **Human-readable cluster label (review and fill):** TODO", "- Representative reviews:"])
        for index in representatives[cluster]:
            item = df.iloc[index]
            lines.append(f"  - `{item['id']}` — {item['text']}")
        lines.append("")
    lines += [
        "## Coherence", "",
        f"- Cosine silhouette score: {silhouette:.4f}" if silhouette is not None
        else "- Cosine silhouette score: not defined for these assignments",
        "- A silhouette score is a heuristic for separation, not proof that cluster labels are correct.",
        "- **Review:** TODO — identify one coherent group and one mixed group.", "",
        "## Near-duplicate detection", "",
        f"- Total flagged pairs at threshold {threshold:.2f}: {len(duplicates)}",
        "- Thresholds need calibration on labeled pairs; semantic similarity alone can produce false positives.", "",
        "| Review A | Review B | Cosine | Excerpts |", "|---|---|---:|---|",
    ]
    for i, j, score in duplicates[:30]:
        lines.append(f"| {safe(df.iloc[i]['id'])} | {safe(df.iloc[j]['id'])} | {score:.4f} | "
                     f"{safe(df.iloc[i]['text'][:65])} / {safe(df.iloc[j]['text'][:65])} |")
    lines += ["", "- **False positive check:** TODO — inspect the list; are any pairs related but not duplicates?",
              "- **False negative check:** TODO — inspect paraphrases missing from the list.",
              "", "## Embedding-based zero-shot label matching", "",
              "We compare each review vector to six **manually written label descriptions** and take the largest cosine score.",
              "This is embedding-prototype classification, **not** a dedicated zero-shot NLI classifier.",
              "Scores are similarities, not calibrated probabilities.", "",
              f"- Example input: {example_text}",
              f"- Predicted label: `{sample_pred[0]}` (similarity={float(np.max(sample_scores)):.4f})",
              f"- Agreement with known synthetic categories: {accuracy:.1%} (in-sample diagnostic only)",
              "- **Examples and errors:** TODO — inspect at least two correct and two incorrect classifications.", "",
              "## Limitations / observations (complete after running)", "",
              "- Dataset is authored for practice; it is NOT genuine customer reviews.",
              "- KMeans k is a chosen hyperparameter. Compare k values before claiming natural groups.",
              "- Negative and positive sentiments may share a topic; clustering does not guarantee sentiment groups.",
              "- A single cosine threshold does not generalize to every domain or language.",
              "- Class-description wording can materially change prototype classification.",
              "- **Personal observations:** TODO — record real findings from your own run.", "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab 5.3 — cluster and analyze customer reviews")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--all", action="store_true", help="Run all applications and save Markdown results")
    actions.add_argument("--cluster", action="store_true", help="Show clusters and representative reviews")
    actions.add_argument("--dedupe", action="store_true", help="Show near-duplicate review pairs")
    actions.add_argument("--classify", action="store_true", help="Classify the supplied review text")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--text", default="The delivery was delayed and tracking never updated")
    args = parser.parse_args()
    try:
        df = load_reviews(args.dataset)
        if not 2 <= args.k < len(df):
            parser.error("--k must be at least 2 and less than the number of reviews")
        if not 0 <= args.threshold <= 1:
            parser.error("--threshold must be between 0 and 1")
        if not args.text.strip():
            parser.error("--text must not be blank")
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    print(f"Loaded {len(df)} synthetic demonstration reviews from {args.dataset}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    start = perf_counter()
    vectors = normalized_embeddings(model, df["text"].tolist())
    duration = perf_counter() - start
    print(f"Embeddings: {vectors.shape}; embedding time: {duration:.2f}s")
    if args.classify:
        class_names = list(CLASS_DESCRIPTIONS)
        descriptions = normalized_embeddings(model, list(CLASS_DESCRIPTIONS.values()))
        sample = normalized_embeddings(model, [args.text])
        predictions, scores = classify_vectors(sample, descriptions, class_names)
        print(f"Text: {args.text}\nPredicted label: {predictions[0]}")
        for name, score in sorted(zip(class_names, scores[0]), key=lambda p: -p[1]):
            print(f"  {name}: cosine={score:.4f}")
    elif args.dedupe:
        pairs = near_duplicate_pairs(vectors, args.threshold)
        print(f"Found {len(pairs)} candidate near-duplicate pairs at threshold {args.threshold:.2f}")
        for i, j, score in pairs[:25]:
            print(f"  {df.iloc[i]['id']} <-> {df.iloc[j]['id']}  cosine={score:.4f}")
            print(f"    {df.iloc[i]['text']}\n    {df.iloc[j]['text']}")
    elif args.cluster:
        assignments, reps, silhouette = cluster_reviews(vectors, args.k)
        print(f"KMeans k={args.k} | cosine silhouette={silhouette}")
        for cluster in range(args.k):
            print(f"\nCluster {cluster} ({sum(assignments == cluster)} examples), proposed label: [review examples]")
            for i in reps[cluster]:
                print(f"  {df.iloc[i]['id']}: {df.iloc[i]['text']}")
    else:
        report = build_report(df, vectors, model, args.k, args.threshold, args.text, duration)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
        print(f"Full report written to: {args.report}")
        print("Open it in VS Code to give clusters human-readable labels and inspect errors.")


if __name__ == "__main__":
    main()
