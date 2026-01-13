from __future__ import annotations

import os
import json
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor, as_completed
import boto3

from test_text_extraction_ import PDFProcessor
from chunker import chunk_text
from embedding_ import milvus_store
from metadata_ import extract_metadata

# --------------------------------------------------
# ENV
# --------------------------------------------------
load_dotenv()

YOLO_MODEL_PATH = os.getenv("YOLO")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "vlsi")

S3_ENDPOINT = os.getenv("S3_ENDPOINT")
S3_BUCKET = os.getenv("S3_BUCKET")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")

MAX_WORKERS = min(os.cpu_count(), 4)

STATE_FILE = "processed_files.json"

# --------------------------------------------------
# S3 CLIENT
# --------------------------------------------------
def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_DEFAULT_REGION,
    )

# --------------------------------------------------
# STATE MANAGEMENT
# --------------------------------------------------
def load_processed_files() -> set:
    if not os.path.exists(STATE_FILE):
        return set()

    with open(STATE_FILE, "r") as f:
        data = json.load(f)

    return set(data.get("processed_files", []))


def save_processed_file(file_key: str):
    processed = load_processed_files()
    processed.add(file_key)

    with open(STATE_FILE, "w") as f:
        json.dump({"processed_files": sorted(processed)}, f, indent=2)

# --------------------------------------------------
# LOAD DOCUMENTS FROM S3
# --------------------------------------------------
def list_s3_pdfs():
    s3 = get_s3_client()
    response = s3.list_objects_v2(Bucket=S3_BUCKET)

    docs = []
    for obj in response.get("Contents", []):
        if obj["Key"].lower().endswith(".pdf"):
            docs.append({
                "bucket": S3_BUCKET,
                "key": obj["Key"],
                "file_name": obj["Key"].split("/")[-1],
            })
    return docs

# --------------------------------------------------
# PER-DOCUMENT WORKER
# --------------------------------------------------
def process_single_document(doc):
    try:
        print(f"[START] {doc['file_name']}")

        s3 = get_s3_client()
        obj = s3.get_object(Bucket=doc["bucket"], Key=doc["key"])
        pdf_bytes = obj["Body"].read()

        processor = PDFProcessor(YOLO_MODEL_PATH)
        raw_text = processor.process_pdf(pdf_bytes)

        if not raw_text.strip():
            return f"[SKIP] Empty PDF: {doc['file_name']}"

        metadata = extract_metadata(raw_text)
        chunks = chunk_text(raw_text)

        if not chunks:
            return f"[SKIP] No chunks: {doc['file_name']}"

        milvus_store(
            collection_name=COLLECTION_NAME,
            chunks=chunks,
            domain=metadata["domain"],
            stage=metadata["stage"],
            type_=metadata["type"],
            version=metadata["version"],
            vendor=metadata["vendor"],
            source=doc["file_name"],
        )

        save_processed_file(doc["key"])

        return f"[DONE] {doc['file_name']}"

    except Exception as e:
        return f"[ERROR] {doc['file_name']} → {str(e)}"

# --------------------------------------------------
# MAIN SCHEDULED RUN
# --------------------------------------------------
def run_scheduled_pipeline():
    processed_files = load_processed_files()
    all_docs = list_s3_pdfs()

    new_docs = [
        d for d in all_docs if d["key"] not in processed_files
    ]

    print(f"📄 Total PDFs in S3      : {len(all_docs)}")
    print(f"✅ Already processed     : {len(processed_files)}")
    print(f"🆕 New PDFs to process   : {len(new_docs)}")

    if not new_docs:
        print("🚫 No new documents found.")
        return

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(process_single_document, doc)
            for doc in new_docs
        ]

        for future in as_completed(futures):
            print(future.result())

# --------------------------------------------------
# ENTRY
# --------------------------------------------------
if __name__ == "__main__":
    run_scheduled_pipeline()
