from pathlib import Path
from typing import List, Dict

import chromadb
from chromadb.config import Settings
from pypdf import PdfReader
from tqdm import tqdm
import ollama

PDF_DIR = Path("data/pdfs")
CHROMA_DIR = "data/chroma"
COLLECTION_NAME = "henry_food_papers"

EMBED_MODEL = "nomic-embed-text"

CHUNK_CHARS = 1800
OVERLAP_CHARS = 300

def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        pages.append(f"\n\n[PAGE {i+1}]\n{txt}")
    return "\n".join(pages)

def chunk_text(text: str, chunk_chars: int = CHUNK_CHARS, overlap_chars: int = OVERLAP_CHARS) -> List[str]:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap_chars)
    return chunks

def embed_texts(texts: List[str]) -> List[List[float]]:
    embs = []
    for t in texts:
        resp = ollama.embeddings(model=EMBED_MODEL, prompt=t)
        embs.append(resp["embedding"])
    return embs

def main():
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {PDF_DIR.resolve()}")

    for pdf_path in tqdm(pdfs, desc="Ingesting PDFs"):
        paper_id = pdf_path.stem
        probe_id = f"{paper_id}_chunk_0"

        # Skip if already ingested
        try:
            got = collection.get(ids=[probe_id])
            if got and got.get("ids"):
                continue
        except Exception:
            pass

        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text)

        ids = [f"{paper_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas: List[Dict] = [
            {"paper_id": paper_id, "source_file": str(pdf_path), "chunk_index": i}
            for i in range(len(chunks))
        ]

        embeddings = embed_texts(chunks)

        collection.add(
            ids=ids,
            documents=chunks,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    print("Done. Vector store saved to:", Path(CHROMA_DIR).resolve())
    print("Collection:", COLLECTION_NAME)
    print("Total chunks:", collection.count())

if __name__ == "__main__":
    main()
