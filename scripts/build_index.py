"""Reproducible command line index builder, run from the project root."""
import argparse
from pathlib import Path
from src.data_loader import DEFAULT_DATASET
from src.embeddings import DEFAULT_MODEL
from src.indexing import DEFAULT_DB, index_documents


def main():
    parser = argparse.ArgumentParser(description='Build/update the Week 5 semantic index')
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--db-dir', type=Path, default=DEFAULT_DB)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--chunk-words', type=int, default=120)
    parser.add_argument('--overlap-words', type=int, default=20)
    args = parser.parse_args()
    info = index_documents(args.dataset, args.db_dir, args.model, args.chunk_words, args.overlap_words)
    for key, val in info.items():
        print(f'{key}: {val}')


if __name__ == '__main__':
    main()
