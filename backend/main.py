import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import get_ollama_status, load_env_files
from backend.pdf_processor import extract_text_from_pdf
from backend.rag_engine import ask_question, build_vector_store

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "data" / "papers"
FRONTEND_DIR = BASE_DIR / "frontend"

load_env_files()

app = FastAPI(title="AI Research Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/health")
async def health_check():
    ollama_status = get_ollama_status()
    return {
        "status": "ok",
        "ollama_available": ollama_status["available"],
        "ollama_model": ollama_status["resolved_model"] or ollama_status["model"],
        "ollama_base_url": ollama_status["base_url"],
        "ollama_detail": ollama_status["detail"],
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    file_path = UPLOAD_DIR / Path(file.filename).name

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = extract_text_from_pdf(str(file_path))
    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="The uploaded PDF does not contain extractable text.",
        )

    try:
        build_vector_store(text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "message": "PDF processed successfully",
        "filename": file_path.name,
    }


@app.post("/ask")
async def ask(question: str):
    question = question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    try:
        answer = ask_question(question)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"answer": answer}


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
