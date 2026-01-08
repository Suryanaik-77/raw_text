import os
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor, as_completed

import sys
sys.path.append('src/mvp_rag')

from text_extraction_ import PDFProcessor
from chunker import chunk_text
from embedding_ import milvus_store
from metadata_ import extract_metadata
from document_loader import loading_docs

load_dotenv()

YOLO_MODEL_PATH = os.getenv("YOLO")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "vlsi_rag")
MAX_WORKERS = min(os.cpu_count(), 4)


def process_single_document(doc: dict):
    try:
        file_path = doc["file_path"]
        file_name = doc["file_name"]

        print(f"[START] {file_name}")

        processor = PDFProcessor(YOLO_MODEL_PATH)

        # 1️⃣ Extract raw text for metadata
        raw_text = processor.process_pdf(file_path)
        if not raw_text.strip():
            return f"[SKIP] Empty PDF: {file_name}"

        # 2️⃣ Metadata extraction (LLM)
        metadata = extract_metadata(raw_text)

        domain  = metadata["domain"]
        stage   = metadata["stage"]
        type_   = metadata["type"]
        version = metadata["version"]
        vendor  = metadata["vendor"]

        # 3️⃣ Chunking
        chunks = chunk_text(raw_text)
        if not chunks:
            return f"[SKIP] No chunks: {file_name}"

        # 4️⃣ Milvus insert (schema-aligned)
        milvus_store(
            collection_name=COLLECTION_NAME,
            chunks=chunks,
            domain=domain,
            stage=stage,
            type_=type_,
            version=version,
            vendor=vendor,
            source=file_name
        )

        return f"[DONE] {file_name}"

    except Exception as e:
        return f"[ERROR] {file_name} → {str(e)}"


def run_parallel_indexing():
    documents = loading_docs()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(process_single_document, doc)
            for doc in documents
        ]

        for future in as_completed(futures):
            print(future.result())


if __name__ == "__main__":
    run_parallel_indexing()
