from pathlib import Path
import shutil

from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.config import get_ollama_status, load_env_files, require_ollama_ready
from backend.pdf_processor import extract_text_from_pdf
from backend.rag_engine import ask_question, build_vector_store

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_DIR = BASE_DIR / "data" / "papers"

load_env_files()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str | None = None


@app.get("/health")
def health() -> dict:
    status = get_ollama_status()
    return {
        "ok": True,
        "ollama_available": status["available"],
        "ollama_model": status["resolved_model"] or status["model"],
        "ollama_requested_model": status["model"],
        "ollama_base_url": status["base_url"],
        "detail": status["detail"],
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Choose a PDF file before trying to process it.")

    if Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    filename = Path(file.filename).name
    file_path = UPLOAD_DIR / filename

    try:
        model_name = require_ollama_ready()

        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        text = extract_text_from_pdf(file_path)
        build_vector_store(text, model_name=model_name)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="The paper could not be processed.") from exc
    finally:
        await file.close()

    return {
        "message": "PDF processed and indexed",
        "filename": filename,
        "ollama_model": model_name,
    }


@app.post("/ask")
async def ask(
    question: str | None = Query(default=None),
    request: QuestionRequest | None = Body(default=None),
) -> dict:
    prompt = (question or (request.question if request else "")).strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Provide a question before asking the assistant.")

    try:
        model_name = require_ollama_ready()
        answer = ask_question(prompt, model_name=model_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="The assistant could not answer the question.") from exc

    return {"answer": answer, "ollama_model": model_name}


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
