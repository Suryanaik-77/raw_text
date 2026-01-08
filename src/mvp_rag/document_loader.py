from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DOCS_DIR = BASE_DIR / "docs"

def loading_docs():
    documents = []

    if not DOCS_DIR.exists():
        raise FileNotFoundError(f"Docs directory not found: {DOCS_DIR}")

    for doc_path in DOCS_DIR.iterdir():
        if doc_path.is_file() and doc_path.suffix.lower() == ".pdf":
            documents.append({
                "file_name": doc_path.name,
                "file_path": str(doc_path)
            })

    return documents
