import json
import re
from collections import Counter
from pathlib import Path

from backend.config import get_ollama_base_url, require_ollama_ready
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import ChatOllama
from ollama import ResponseError

BASE_DIR = Path(__file__).resolve().parent.parent
RAG_DIR = BASE_DIR / "data" / "rag"
CHUNKS_DB_PATH = RAG_DIR / "chunks.json"


def _tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def _save_chunks(chunks):
    RAG_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_DB_PATH.write_text(
        json.dumps({"chunks": chunks}, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def _load_chunks():
    if not CHUNKS_DB_PATH.exists():
        raise FileNotFoundError("Upload a PDF before asking questions.")

    payload = json.loads(CHUNKS_DB_PATH.read_text(encoding="utf-8"))
    chunks = payload.get("chunks", [])
    if not chunks:
        raise FileNotFoundError("Upload a PDF before asking questions.")

    return chunks


def _rank_chunks(question, chunks, limit=3):
    question_tokens = _tokenize(question)
    if not question_tokens:
        return chunks[:limit]

    question_counts = Counter(question_tokens)
    scored_chunks = []

    for index, chunk in enumerate(chunks):
        chunk_counts = Counter(_tokenize(chunk))
        overlap_score = sum(chunk_counts[token] * weight for token, weight in question_counts.items())
        unique_hits = sum(1 for token in question_counts if token in chunk_counts)
        phrase_bonus = 3 if question.lower() in chunk.lower() else 0
        score = overlap_score + (unique_hits * 2) + phrase_bonus
        scored_chunks.append((score, -index, chunk))

    scored_chunks.sort(reverse=True)
    best_chunks = [chunk for score, _, chunk in scored_chunks if score > 0][:limit]
    if best_chunks:
        return best_chunks

    return chunks[:limit]


def build_vector_store(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_text(text)
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    if not chunks:
        raise ValueError("The uploaded PDF did not produce usable text chunks.")

    _save_chunks(chunks)

    return {"chunks_indexed": len(chunks)}


def ask_question(question):
    resolved_model = require_ollama_ready()
    chunks = _load_chunks()
    context = "\n\n".join(_rank_chunks(question, chunks, limit=3))

    prompt = f"""
Use the following research paper context to answer the question.

If the context is incomplete, say so clearly instead of inventing facts.

Context:
{context}

Question:
{question}
"""

    llm = ChatOllama(
        model=resolved_model,
        base_url=get_ollama_base_url(),
        temperature=0,
    )

    try:
        response = llm.invoke(prompt)
    except ResponseError as exc:
        detail = str(exc)
        if "requires more system memory" in detail:
            raise RuntimeError(
                f'The Ollama model "{resolved_model}" needs more RAM than is currently available. '
                "Use a smaller local model or free memory, then try again."
            ) from exc
        raise RuntimeError(f"Ollama request failed: {detail}") from exc

    return response.content if hasattr(response, "content") else str(response)
