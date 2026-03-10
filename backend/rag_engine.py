from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from ollama import ResponseError

from backend.config import get_ollama_base_url, require_ollama_ready

BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_DB_DIR = BASE_DIR / "vector_db"


def _resolve_model(model_name: str | None = None) -> str:
    return model_name or require_ollama_ready()


def _build_embeddings(model_name: str) -> OllamaEmbeddings:
    return OllamaEmbeddings(model=model_name, base_url=get_ollama_base_url())


def _translate_ollama_error(exc: ResponseError, model_name: str) -> RuntimeError:
    detail = str(exc)
    if "requires more system memory" in detail:
        return RuntimeError(
            f'The Ollama model "{model_name}" needs more RAM than is currently available. '
            "Use a smaller local model or free memory, then try again."
        )

    return RuntimeError(f"Ollama request failed: {detail}")


def build_vector_store(text: str, model_name: str | None = None) -> int:
    if not text.strip():
        raise RuntimeError("The uploaded PDF did not contain readable text.")

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_text(text)
    if not chunks:
        raise RuntimeError("The uploaded PDF did not contain readable text.")

    resolved_model = _resolve_model(model_name)
    embeddings = _build_embeddings(resolved_model)

    try:
        vector_store = FAISS.from_texts(chunks, embeddings)
    except ResponseError as exc:
        raise _translate_ollama_error(exc, resolved_model) from exc

    vector_store.save_local(str(VECTOR_DB_DIR))
    return len(chunks)


def ask_question(question: str, model_name: str | None = None) -> str:
    if not VECTOR_DB_DIR.exists():
        raise FileNotFoundError("Upload and process a paper before asking questions.")

    resolved_model = _resolve_model(model_name)
    embeddings = _build_embeddings(resolved_model)

    try:
        vector_store = FAISS.load_local(
            str(VECTOR_DB_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError("Upload and process a paper before asking questions.") from exc

    try:
        docs = vector_store.similarity_search(question, k=3)
    except ResponseError as exc:
        raise _translate_ollama_error(exc, resolved_model) from exc

    if not docs:
        raise RuntimeError("The indexed paper did not return any matching context.")

    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = f"""
Answer the question using the research paper context.

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
        raise _translate_ollama_error(exc, resolved_model) from exc

    return response.content if hasattr(response, "content") else str(response)
