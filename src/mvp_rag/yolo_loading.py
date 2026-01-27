from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import threading
from dotenv import load_dotenv
# Thread-safe singleton
_model = None
_lock = threading.Lock()
import os
os.environ["HF_HUB_TIMEOUT"] = "60"

def get_yolo11m():
    global _model
    if _model is None:
        path = hf_hub_download(
            repo_id="Armaggheddon/yolo11-document-layout",
            filename="yolo11m_doc_layout.pt",
        )
        _model = YOLO(path)
        print("✅ YOLO11m loaded")
    return _model

if __name__ == "__main__":
    model = get_yolo11m()
