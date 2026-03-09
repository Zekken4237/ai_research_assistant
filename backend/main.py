from fastapi import FastAPI, UploadFile, File
import shutil
import os

from backend.pdf_processor import extract_text_from_pdf
from backend.rag_engine import build_vector_store
from backend.rag_engine import ask_question 

app = FastAPI()

UPLOAD_DIR = "data/papers"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    file_path = f"{UPLOAD_DIR}/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = extract_text_from_pdf(file_path)

    build_vector_store(text)

    return {"message": "PDF processed successfully"}

@app.post("/ask")
async def ask(question: str):

    answer = ask_question(question)

    return {"answer": answer}
    