from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Iterable, List


def normalize_block(text: str) -> str:
    cleaned = text.replace("\r", "\n")
    while "\n\n\n" in cleaned:
        cleaned = cleaned.replace("\n\n\n", "\n\n")
    return " ".join(cleaned.split()).strip()


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)



