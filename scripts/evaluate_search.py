"""Evaluate retrieval using a small held-out set of *demonstration* relevance labels."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
from src.data_loader import ROOT
from src.search import search_documents
from src.indexing import open_collection, model_collection_name
from src.embeddings import DEFAULT_MODEL

DEFAULT_EVAL = ROOT / 'data' / 'evaluation' / 'final_relevance_queries.csv'
DEFAULT_REPORT = ROOT / 'docs' / 'final_evaluation.md'


def score_retrieval(expected: set[str], ranked_ids: list[str]) -> dict:
    if not expected:
        raise ValueError('Expected relevance labels may not be empty')
    # Score document IDs, not chunk IDs, to avoid duplicate chunks inflating success.
    retrieved = set(ranked_ids)
    return {'hit': int(bool(expected & retrieved)),
            'recall': len(expected & retrieved) / len(expected)}


def evaluate(path: Path = DEFAULT_EVAL, report: Path = DEFAULT_REPORT, top_k: int = 5):
    with path.open(encoding='utf-8-sig', newline='') as f:
        cases = list(csv.DictReader(f))
    if not cases:
        raise ValueError('No evaluation cases supplied')
    collection = open_collection(collection_name=model_collection_name(DEFAULT_MODEL, 120, 20))
    if collection.count() == 0:
        raise RuntimeError('No index found. First run: python -m scripts.build_index')
    lines = ['# Final Retrieval Evaluation', '',
             '> These are hand-created demonstration queries and relevance labels, not a validated benchmark.', '',
             f'- Top-k: {top_k}', f'- Number of held-out test queries: {len(cases)}', '',
             '| ID | Query | Hit@k | Recall@k | Retrieved document IDs |',
             '|---|---|---:|---:|---|']
    scores = []
    for case in cases:
        expected = set(case['relevant_ids'].split(';'))
        results = search_documents(case['query'], top_k=top_k)
        ranked_ids = list(dict.fromkeys(hit['document_id'] for hit in results))
        score = score_retrieval(expected, ranked_ids)
        scores.append(score)
        clean_query = case['query'].replace('|', '/').replace('\n', ' ')
        lines.append(f"| {case['query_id']} | {clean_query} | {score['hit']} | "
                     f"{score['recall']:.3f} | {', '.join(ranked_ids)} |")
    hit_rate = sum(s['hit'] for s in scores) / len(scores)
    recall = sum(s['recall'] for s in scores) / len(scores)
    lines += ['', f'- Mean Hit@{top_k}: **{hit_rate:.3f}**',
              f'- Mean Recall@{top_k}: **{recall:.3f}**', '',
              '## Observations', '', '- TODO: Inspect incorrect and missing results manually.',
              '- TODO: Repeat evaluation after switching to the real corpus with new relevance labels.', '']
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Evaluated {len(cases)} queries. Hit@{top_k}={hit_rate:.3f}, Recall@{top_k}={recall:.3f}')
    print(f'Report: {report}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--queries', type=Path, default=DEFAULT_EVAL)
    parser.add_argument('--output', type=Path, default=DEFAULT_REPORT)
    parser.add_argument('--top-k', type=int, default=5)
    args = parser.parse_args()
    evaluate(args.queries, args.output, args.top_k)
