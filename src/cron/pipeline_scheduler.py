from __future__ import annotations

import os
import json
import boto3
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.mvp_rag.test_text_extraction_ import PDFProcessor
from src.mvp_rag.chunker import chunk_text
from src.mvp_rag.embedding_ import milvus_store
from src.mvp_rag.metadata_ import extract_metadata

# --------------------------------------------------
# ENV
# --------------------------------------------------
load_dotenv()

STATE_FILE = "/app/state/processed_files.json"
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION1", "vlsi")

S3_BUCKET = os.getenv("S3_BUCKET")
S3_ENDPOINT = os.getenv("S3_ENDPOINT")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION")

YOLO_MODEL_PATH = os.getenv("YOLO")

MAX_WORKERS = int(os.getenv("CRON_WORKERS", "3"))

# --------------------------------------------------
# State helpers
# --------------------------------------------------
def load_state() -> set:
    if not os.path.exists(STATE_FILE):
        return set()
    with open(STATE_FILE, "r") as f:
        return set(json.load(f))


def save_state(processed: set):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(sorted(processed), f, indent=2)

# --------------------------------------------------
# S3 client
# --------------------------------------------------
def get_s3():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        region_name=AWS_REGION,
    )

# --------------------------------------------------
# Worker (single PDF)
# --------------------------------------------------
def process_one_pdf(key: str, s3, processor) -> str | None:
    try:
        print(f"[NEW] {key}")

        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
        pdf_bytes = obj["Body"].read()

        raw_text = processor.process_pdf(pdf_bytes)
        if not raw_text.strip():
            print(f"[SKIP] Empty PDF → {key}")
            return key

        metadata = extract_metadata(raw_text)
        chunks = chunk_text(raw_text)

        if not chunks:
            print(f"[SKIP] No chunks → {key}")
            return key

        milvus_store(
            collection_name=COLLECTION_NAME,
            chunks=chunks,
            domain=metadata["domain"],
            stage=metadata["stage"],
            type_=metadata["type"],
            version=metadata["version"],
            vendor=metadata["vendor"],
            source=key,
        )

        print(f"[DONE] {key}")
        return key

    except Exception as e:
        print(f"[ERROR] {key} → {e}")
        return None

# --------------------------------------------------
# Main cron task
# --------------------------------------------------
def run():
    print("⏱ Cron ingestion started")

    processed = load_state()
    s3 = get_s3()

    response = s3.list_objects_v2(Bucket=S3_BUCKET)
    objects = response.get("Contents", [])

    new_keys = [
        obj["Key"]
        for obj in objects
        if obj["Key"].lower().endswith(".pdf")
        and obj["Key"] not in processed
    ]

    if not new_keys:
        print("✅ No new PDFs found")
        return

    print(f"📄 Found {len(new_keys)} new PDFs")

    processor = PDFProcessor(YOLO_MODEL_PATH)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_one_pdf, key, s3, processor): key
            for key in new_keys
        }

        for future in as_completed(futures):
            result = future.result()
            if result:
                processed.add(result)

    save_state(processed)
    print("✅ Cron ingestion finished")

# --------------------------------------------------
# Entry
# --------------------------------------------------
if __name__ == "__main__":
    run()
