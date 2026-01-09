import json
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import ollama
import chromadb
from chromadb.config import Settings

# -----------------------------
# Config
# -----------------------------
CHROMA_DIR = "data/chroma"
COLLECTION_NAME = "henry_food_papers"

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:3b-instruct"

TOP_K = 6
MAX_CONTEXT_CHARS = 12000  # keep prompts reasonable on your i5

DB_PATH = "data/querylog.sqlite"

SYSTEM_PROMPT = """You are a careful research assistant.
Use ONLY the provided excerpts. If they are insufficient, say what's missing.
Cite sources as [paper_id chunk_index]. Do not invent facts.
Keep answers practical and concise.
"""

# -----------------------------
# Persistence (SQLite)
# -----------------------------
def init_db(db_path: str = DB_PATH) -> None:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER NOT NULL,
            question TEXT NOT NULL,
            embed_model TEXT NOT NULL,
            embedding_json TEXT NOT NULL
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER NOT NULL,
            query_id INTEGER NOT NULL,
            llm_model TEXT NOT NULL,
            top_k INTEGER NOT NULL,
            system_prompt TEXT NOT NULL,
            user_prompt TEXT NOT NULL,
            response TEXT NOT NULL,
            retrieval_json TEXT NOT NULL,
            FOREIGN KEY(query_id) REFERENCES queries(id)
        )
        """)
        # Optional feedback for RL/analysis later
        conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER NOT NULL,
            run_id INTEGER NOT NULL,
            rating INTEGER,               -- e.g. -1/0/1 or 1-5
            notes TEXT,
            FOREIGN KEY(run_id) REFERENCES runs(id)
        )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_queries_ts ON queries(ts)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_ts ON runs(ts)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_query_id ON runs(query_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_run_id ON feedback(run_id)")
        conn.commit()

def save_query(question: str, embed_model: str, embedding: List[float], db_path: str = DB_PATH) -> int:
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO queries (ts, question, embed_model, embedding_json) VALUES (?, ?, ?, ?)",
            (int(time.time()), question, embed_model, json.dumps(embedding)),
        )
        conn.commit()
        return int(cur.lastrowid)

def save_run(
    query_id: int,
    llm_model: str,
    top_k: int,
    system_prompt: str,
    user_prompt: str,
    response: str,
    retrieval: Dict[str, Any],
    db_path: str = DB_PATH,
) -> int:
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO runs
               (ts, query_id, llm_model, top_k, system_prompt, user_prompt, response, retrieval_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (int(time.time()), query_id, llm_model, top_k, system_prompt, user_prompt, response, json.dumps(retrieval)),
        )
        conn.commit()
        return int(cur.lastrowid)

def save_feedback(run_id: int, rating: Optional[int], notes: str = "", db_path: str = DB_PATH) -> int:
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO feedback (ts, run_id, rating, notes) VALUES (?, ?, ?, ?)",
            (int(time.time()), run_id, rating, notes),
        )
        conn.commit()
        return int(cur.lastrowid)

# -----------------------------
# RAG steps
# -----------------------------
def embed_question(question: str, model: str = EMBED_MODEL) -> List[float]:
    resp = ollama.embeddings(model=model, prompt=question)
    return resp["embedding"]

def get_collection(chroma_dir: str = CHROMA_DIR, name: str = COLLECTION_NAME):
    client = chromadb.PersistentClient(
        path=chroma_dir,
        settings=Settings(anonymized_telemetry=False)
    )
    return client.get_collection(name)

def retrieve_chunks(
    query_embedding: List[float],
    top_k: int = TOP_K,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    col = get_collection()
    res = col.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas"]
    )
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    return docs, metas

def format_context(docs: List[str], metas: List[Dict[str, Any]], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    parts = []
    total = 0
    for doc, m in zip(docs, metas):
        block = f"[{m.get('paper_id','?')} {m.get('chunk_index','?')}]\n{doc}".strip()
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block) + 5
    return "\n\n---\n\n".join(parts)

def build_user_prompt(question: str, docs: List[str], metas: List[Dict[str, Any]]) -> str:
    context = format_context(docs, metas)
    return f"""Context excerpts:
{context}

Question: {question}

Answer using ONLY the excerpts. Include citations like [paper_id chunk_index]."""

def generate_answer(user_prompt: str, system_prompt: str = SYSTEM_PROMPT, model: str = LLM_MODEL) -> str:
    out = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return out["message"]["content"]

# -----------------------------
# Orchestrator
# -----------------------------
def rag_answer(question: str) -> Dict[str, Any]:
    q_emb = embed_question(question, EMBED_MODEL)
    query_id = save_query(question, EMBED_MODEL, q_emb)

    docs, metas = retrieve_chunks(q_emb, TOP_K)
    user_prompt = build_user_prompt(question, docs, metas)
    response = generate_answer(user_prompt, SYSTEM_PROMPT, LLM_MODEL)

    retrieval_payload = {
        "top_k": TOP_K,
        "chunks": [
            {
                "paper_id": m.get("paper_id"),
                "chunk_index": m.get("chunk_index"),
                "source_file": m.get("source_file"),
                "excerpt_preview": (d[:240] + "…") if len(d) > 240 else d,
            }
            for d, m in zip(docs, metas)
        ],
    }

    run_id = save_run(
        query_id=query_id,
        llm_model=LLM_MODEL,
        top_k=TOP_K,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response=response,
        retrieval=retrieval_payload,
    )

    return {
        "query_id": query_id,
        "run_id": run_id,
        "response": response,
        "retrieval": retrieval_payload,
    }

def main():
    print("Local RAG (Ollama + Chroma). Type 'exit' to quit.")
    print(f"LLM={LLM_MODEL} | EMBED={EMBED_MODEL} | top_k={TOP_K}")
    while True:
        q = input("\nQuestion: ").strip()
        if not q or q.lower() in {"exit", "quit"}:
            break

        result = rag_answer(q)
        print(f"\n(run_id={result['run_id']} query_id={result['query_id']})")
        print(result["response"])

        # Optional: quick feedback logging for RL/analysis later
        fb = input("\nFeedback? (+1 good / 0 meh / -1 bad / Enter skip): ").strip()
        if fb in {"+1", "1", "0", "-1"}:
            rating = int(fb)
            notes = input("Notes (optional): ").strip()
            save_feedback(result["run_id"], rating, notes)
            print("Saved feedback.")

if __name__ == "__main__":
    main()
