from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Iterable, List


def normalize_block(text: str) -> str:
    cleaned = text.replace("\r", "\n")
    while "\n\n\n" in cleaned:
        cleaned = cleaned.replace("\n\n\n", "\n\n")
    return " ".join(cleaned.split()).strip()


def chunk_text(
    text: str,
    chunk_size: int = 1500,
    chunk_overlap: int = 300
) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    text = normalize_block(text)
    return splitter.split_text(text)

with open("test/dcug 1.txt","r",encoding="utf-8") as f:
    text = f.read()

print(chunk_text(text)[0])

