from pathlib import Path
from typing import List, Dict
import time

import chromadb
from chromadb.config import Settings
from pypdf import PdfReader
from tqdm import tqdm
import ollama

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "data/pdfs"
CHROMA_DIR = str(BASE_DIR / "data/chroma")
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
    for i, t in enumerate(tqdm(texts, desc="    embedding chunks", unit="chunk", leave=False)):
        resp = ollama.embeddings(model=EMBED_MODEL, prompt=t)
        embs.append(resp["embedding"])
        # periodic heartbeat in case tqdm isn't visible for some reason
        if (i + 1) % 25 == 0:
            print(f"    embedded {i+1}/{len(texts)} chunks")
    return embs

def already_ingested(collection, paper_id: str) -> bool:
    # We consider a paper ingested if chunk_0 exists
    probe_id = f"{paper_id}_chunk_0"
    try:
        got = collection.get(ids=[probe_id])
        return bool(got and got.get("ids"))
    except Exception:
        return False

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

    print(f"Found {len(pdfs)} PDFs in {PDF_DIR}")
    processed = 0
    skipped = 0
    empty_text = 0

    for idx, pdf_path in enumerate(pdfs, start=1):
        paper_id = pdf_path.stem

        if already_ingested(collection, paper_id):
            print(f"[{idx}/{len(pdfs)}] SKIP (already ingested): {pdf_path.name}")
            skipped += 1
            continue

        print(f"\n[{idx}/{len(pdfs)}] PROCESS: {pdf_path.name}")
        t0 = time.time()
        text = extract_text_from_pdf(pdf_path)
        t1 = time.time()

        text_len = len(text.strip())
        print(f"  extracted in {t1 - t0:.1f}s | text chars: {text_len}")

        if text_len < 200:
            print("  WARNING: almost no text extracted (likely scanned PDF). Skipping for now.")
            empty_text += 1
            continue

        chunks = chunk_text(text)
        print(f"  chunks: {len(chunks)} (chunk_chars={CHUNK_CHARS}, overlap={OVERLAP_CHARS})")

        t2 = time.time()
        embeddings = embed_texts(chunks)
        t3 = time.time()
        print(f"  embedded in {t3 - t2:.1f}s")

        ids = [f"{paper_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas: List[Dict] = [
            {"paper_id": paper_id, "source_file": str(pdf_path), "chunk_index": i}
            for i in range(len(chunks))
        ]

        collection.add(
            ids=ids,
            documents=chunks,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        processed += 1
        print(f"  saved to chroma | total chunks now: {collection.count()}")

    print("\n=== SUMMARY ===")
    print("Processed:", processed)
    print("Skipped already ingested:", skipped)
    print("Skipped (no text extracted):", empty_text)
    print("Vector store:", CHROMA_DIR)
    print("Collection:", COLLECTION_NAME)
    print("Total chunks:", collection.count())

if __name__ == "__main__":
    main()
