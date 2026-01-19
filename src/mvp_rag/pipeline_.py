import os
import sys
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append("src/mvp_rag")

from text_extraction_ import PDFProcessor
from chunker import chunk_text
from embedding_ import milvus_store
from metadata_ import extract_metadata
from document_loader import loading_docs

load_dotenv()

YOLO_MODEL_PATH = os.getenv("YOLO")
MAX_WORKERS = min(os.cpu_count() or 1, 4)


def process_single_document(doc: dict):
    file_path = doc.get("file_path")
    file_name = doc.get("file_name")

    try:
        print(f"[START] {file_name}")

        processor = PDFProcessor(YOLO_MODEL_PATH)
        raw_text = processor.process_pdf(file_path)

        if not raw_text or not raw_text.strip():
            return f"[SKIP] Empty PDF: {file_name}"

        metadata = extract_metadata(raw_text)
        print(metadata)
        domain = metadata.get("domain", "default")
        stage = metadata.get("stage", "unknown")
        doc_type = metadata.get("type", "unknown")
        version = metadata.get("version", "unknown")
        vendor = metadata.get("vendor", "unknown")
        tool = metadata.get("Tool", "unknown")

        collection_name = domain.replace(" ","_")
        tool = tool.replace(" ","_")
        chunks = chunk_text(raw_text)
        if not chunks:
            return f"[SKIP] No chunks: {file_name}"

        milvus_store(
            collection_name=collection_name,
            chunks=chunks,
            domain=domain,
            stage=stage,
            type_=doc_type,
            version=version,
            vendor=vendor,
            source=file_name,
            tool=tool
        )

        return f"[DONE] {file_name}"

    except Exception as e:
        return f"[ERROR] {file_name} → {str(e)}"


def run_parallel_indexing():
    documents = loading_docs()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_single_document, doc) for doc in documents]
        for future in as_completed(futures):
            print(future.result())


if __name__ == "__main__":
    run_parallel_indexing()
