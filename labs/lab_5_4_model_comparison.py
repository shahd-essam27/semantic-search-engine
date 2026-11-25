"""Week 5 / Milestone 5 / Lab 5.4: English vs multilingual retrieval.

Run from project root:
  python -m labs.lab_5_4_model_comparison --evaluate
  python -m labs.lab_5_4_model_comparison --evaluate --device cpu

Corpus stays identical; each model encodes both corpus and queries independently.
All metrics in reports are measured on your computer, not prefilled.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / 'data' / 'raw' / 'documents.csv'
QUERIES = ROOT / 'data' / 'evaluation' / 'model_comparison_queries.csv'
REPORT = ROOT / 'docs' / 'model_comparison.md'
DETAILS = ROOT / 'docs' / 'model_comparison_details.csv'
MODELS = {
    'english': 'sentence-transformers/all-MiniLM-L6-v2',
    'multilingual': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
}
K = 5


def read_documents(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as fh:
        rows = list(csv.DictReader(fh))
    if not rows or not {'id', 'text'}.issubset(rows[0]):
        raise ValueError('documents.csv must contain id and text columns and at least one row')
    if any(not x['id'].strip() or not x['text'].strip() for x in rows):
        raise ValueError('Blank id/text in documents.csv')
    if len({x['id'] for x in rows}) != len(rows):
        raise ValueError('Duplicate document IDs')
    return rows


def read_queries(path: Path, doc_ids: set[str]) -> list[dict[str, Any]]:
    with path.open('r', encoding='utf-8-sig', newline='') as fh:
        data = list(csv.DictReader(fh))
    required = {'query_id', 'language', 'query', 'relevant_ids', 'note'}
    if not data or not required.issubset(data[0]):
        raise ValueError(f'Evaluation CSV requires columns {sorted(required)}')
    if len(set(x['query_id'] for x in data)) != len(data):
        raise ValueError('Duplicate query IDs')
    out = []
    for x in data:
        gold = [v.strip() for v in x['relevant_ids'].split(';') if v.strip()]
        if not x['query'].strip() or not gold:
            raise ValueError(f'Empty query or gold IDs: {x["query_id"]}')
        missing = set(gold) - doc_ids
        if missing:
            raise ValueError(f'Unknown relevant IDs for {x["query_id"]}: {sorted(missing)}')
        if x['language'] not in {'en', 'ar'}:
            raise ValueError('language must be en or ar')
        out.append({**x, 'gold': gold})
    return out


def normalize(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError('Expected 2D vectors')
    denom = np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)
    return arr / denom


def topk(doc_ids: list[str], corpus_vectors: np.ndarray, query_vector: np.ndarray, k: int = K) -> tuple[list[str], list[float]]:
    if corpus_vectors.shape[0] != len(doc_ids):
        raise ValueError('Vector count != document count')
    if corpus_vectors.shape[1] != len(query_vector):
        raise ValueError('Dimension mismatch (never mix embedding models)')
    similarities = corpus_vectors @ np.asarray(query_vector, dtype=np.float32)
    indices = np.argsort(-similarities, kind='stable')[:min(k, len(doc_ids))]
    return [doc_ids[i] for i in indices], [float(similarities[i]) for i in indices]


def recall_at_k(predicted: list[str], relevant: list[str]) -> float:
    """Fraction of all labeled-relevant IDs that appear among top-k predictions."""
    gold = set(relevant)
    if not gold:
        raise ValueError('Relevant IDs cannot be empty')
    return len(gold.intersection(predicted)) / len(gold)


def hit_at_k(predicted: list[str], relevant: list[str]) -> float:
    return float(bool(set(predicted).intersection(relevant)))


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float('nan')


def model_disk_mb(model_name: str) -> str:
    """Estimate size of selected weights in local Hugging Face cache; N/A when unknown."""
    try:
        from huggingface_hub import snapshot_download
        folder = Path(snapshot_download(repo_id=model_name, local_files_only=True))
        files = list(folder.rglob('*.safetensors'))
        if not files:
            files = list(folder.rglob('pytorch_model.bin'))
        if not files:
            return 'N/A'
        return f'{sum(p.stat().st_size for p in files) / 1024**2:.1f}'
    except Exception:
        return 'N/A'


def rss_mb() -> float | None:
    """Process resident-set memory when optional psutil happens to be installed."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024**2
    except (ImportError, OSError):
        return None


def run_model(model_key: str, model_name: str, docs: list[dict[str, str]], queries: list[dict[str, Any]], device: str) -> tuple[dict, list[dict]]:
    from sentence_transformers import SentenceTransformer
    start = perf_counter()
    model = SentenceTransformer(model_name, device=device)
    load_time = perf_counter() - start
    baseline_memory = rss_mb()
    start = perf_counter()
    doc_embeddings = normalize(model.encode([d['text'] for d in docs], batch_size=32, convert_to_numpy=True, show_progress_bar=True))
    corpus_seconds = perf_counter() - start
    after_memory = rss_mb()
    ids = [d['id'] for d in docs]
    details = []
    for q in queries:
        start = perf_counter()
        embedding = normalize(model.encode([q['query']], convert_to_numpy=True))[0]
        found, scores = topk(ids, doc_embeddings, embedding)
        milliseconds = (perf_counter() - start) * 1000
        details.append({
            'model': model_key, 'query_id': q['query_id'], 'language': q['language'],
            'query': q['query'], 'gold_ids': ';'.join(q['gold']),
            'top5_ids': ';'.join(found), 'top5_scores': ';'.join(f'{s:.4f}' for s in scores),
            'recall_at_5': recall_at_k(found, q['gold']), 'hit_at_5': hit_at_k(found, q['gold']),
            'query_ms': milliseconds,
        })
    summary = {
        'model': model_key, 'model_name': model_name, 'dimension': int(doc_embeddings.shape[1]),
        'model_weights_mb': model_disk_mb(model_name), 'model_load_s': load_time,
        'corpus_embed_s': corpus_seconds, 'query_avg_ms': mean([x['query_ms'] for x in details]),
        'rss_before_mb': baseline_memory, 'rss_after_mb': after_memory,
        'english_recall': mean([x['recall_at_5'] for x in details if x['language']=='en']),
        'arabic_recall': mean([x['recall_at_5'] for x in details if x['language']=='ar']),
        'english_hit': mean([x['hit_at_5'] for x in details if x['language']=='en']),
        'arabic_hit': mean([x['hit_at_5'] for x in details if x['language']=='ar']),
        'overall_recall': mean([x['recall_at_5'] for x in details]),
        'device': str(model.device),
    }
    del model, doc_embeddings
    return summary, details


def markdown_report(summaries: list[dict], details: list[dict], docs_count: int, queries: list[dict]) -> str:
    lines = [
        '# Lab 5.4 — English vs Multilingual Embedding Comparison', '',
        f'Generated: {datetime.now().astimezone().isoformat(timespec="seconds")}', '',
        '## Setup and evaluation method',
        f'- Corpus: `data/raw/documents.csv` ({docs_count} records); all documents are **English**.',
        f'- Queries: `data/evaluation/model_comparison_queries.csv` ({len(queries)} queries; English and Arabic).',
        '- Both models encode **the same documents and queries**; each model creates its **own** vector space.',
        '- Top-k: 5. Cosine similarity of normalized vectors.',
        '- Recall@5 = relevant labeled document IDs retrieved / all labeled relevant IDs.',
        '- Hit@5 = 1 when at least one labeled relevant ID appears in top five.',
        '- Labels are a small, manually authored teaching set; they are **not exhaustive relevance judgments**.',
        '- Arabic tests are **cross-language retrieval** (Arabic queries over English documents); this does not evaluate an Arabic-language document collection.',
        '- Query time includes query embedding and cosine ranking; excludes disk access and model loading.',
        '- Timing is device-dependent; first query may include warm-up overhead.',
        '- RSS is whole-process resident memory, *not* model-only memory; N/A if optional psutil is not available.', '',
        '## Measured comparison', '',
        '| Metric | English-focused | Multilingual |', '|---|---:|---:|',
    ]
    keys = [('Model','model_name'),('Embedding dimension','dimension'),('Cached weight files (MB)','model_weights_mb'),
            ('Device','device'),('Model load (s)','model_load_s'),('Corpus embedding (s)','corpus_embed_s'),
            ('Average query time (ms)','query_avg_ms'),('Process RSS before embedding (MB)','rss_before_mb'),
            ('Process RSS after embedding (MB)','rss_after_mb'),('English Recall@5','english_recall'),
            ('Arabic Recall@5','arabic_recall'),('English Hit@5','english_hit'),('Arabic Hit@5','arabic_hit'),
            ('Overall Recall@5','overall_recall')]
    def val(s: dict, key: str) -> str:
        v = s[key]
        return 'N/A' if v is None else (f'{v:.3f}' if isinstance(v,float) else str(v))
    for label, key in keys:
        lines.append(f'| {label} | {val(summaries[0],key)} | {val(summaries[1],key)} |')
    lines.extend(['', '## Representative successes and failures', '',
                  'Each row shows retrieved document IDs. Consult `model_comparison_details.csv` for scores and all queries.', '',
                  '| Model | Query | Language | Recall@5 | Top-5 IDs |', '|---|---|---|---:|---|'])
    for key in ['english','multilingual']:
        rows = [r for r in details if r['model']==key]
        # Show one strongest and one weakest example from each language when present.
        for lang in ['en','ar']:
            subset = [r for r in rows if r['language']==lang]
            for r in ([max(subset, key=lambda x:x['recall_at_5']), min(subset, key=lambda x:x['recall_at_5'])] if subset else []):
                lines.append(f"| {key} | {r['query_id']} | {lang} | {r['recall_at_5']:.3f} | {r['top5_ids'].replace(';', ', ')} |")
    lines.extend(['', '## Your interpretation (complete after running)', '',
                  '- **Best model for English queries:** TODO — justify using English Recall@5 and speed.',
                  '- **Best model for Arabic cross-language queries:** TODO — justify using Arabic Recall@5.',
                  '- **Final model choice for Milestone 6:** TODO — explain corpus language, quality, latency, and size trade-offs.',
                  '- **Failure analysis:** TODO — inspect two low-scoring queries from `model_comparison_details.csv`.',
                  '- **Limitations:** The synthetic English-only corpus and small manually authored relevance set can bias results; test on a real corpus before making a portfolio claim.',
                  ''])
    return '\n'.join(lines)


def evaluate(corpus: Path, eval_csv: Path, report: Path, details_csv: Path, device: str) -> None:
    docs = read_documents(corpus)
    queries = read_queries(eval_csv, {d['id'] for d in docs})
    print(f'Corpus={len(docs)} records, queries={len(queries)}, device={device}')
    summaries, all_details = [], []
    for key, name in MODELS.items():
        print(f'\nEvaluating {key}: {name}')
        summary, details = run_model(key, name, docs, queries, device)
        summaries.append(summary)
        all_details.extend(details)
        print(f"  dimension={summary['dimension']}  English R@5={summary['english_recall']:.3f}  Arabic R@5={summary['arabic_recall']:.3f}")
    report.parent.mkdir(parents=True,exist_ok=True)
    details_csv.parent.mkdir(parents=True,exist_ok=True)
    with details_csv.open('w', encoding='utf-8-sig', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_details[0]))
        writer.writeheader()
        writer.writerows(all_details)
    report.write_text(markdown_report(summaries, all_details, len(docs), queries), encoding='utf-8')
    print(f'\nSaved report: {report}\nSaved query details: {details_csv}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluate', action='store_true', help='Run both models on bilingual labeled queries')
    parser.add_argument('--device', default='cpu', choices=['cpu','cuda'], help='Use cpu for consistent measurements')
    parser.add_argument('--corpus', type=Path, default=CORPUS)
    parser.add_argument('--queries', type=Path, default=QUERIES)
    parser.add_argument('--report', type=Path, default=REPORT)
    parser.add_argument('--details', type=Path, default=DETAILS)
    args = parser.parse_args()
    if not args.evaluate:
        parser.print_help()
        return
    evaluate(args.corpus,args.queries,args.report,args.details,args.device)


if __name__ == '__main__':
    main()
