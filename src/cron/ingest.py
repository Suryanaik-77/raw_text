"""
Manual ingestion script for RAG pipeline.

This script:
- Loads PDFs from docs/ (via document_loader)
- Runs parallel PDF processing
- Stores embeddings in Milvus

Usage:
    python ingest_docs.py
"""

import os
import argparse
from dotenv import load_dotenv

# Import your pipeline runner
from mvp_rag.test_pipeline import run_parallel_indexing

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run RAG PDF ingestion")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override number of parallel workers"
    )

    args = parser.parse_args()

    if args.workers:
        os.environ["MAX_WORKERS_OVERRIDE"] = str(args.workers)

    print("🚀 Starting RAG ingestion pipeline")
    run_parallel_indexing()
    print("✅ Ingestion completed")

if __name__ == "__main__":
    main()
