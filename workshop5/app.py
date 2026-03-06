import os
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

load_dotenv()

app = FastAPI()

# ---------- Initialize Global Components ----------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

retriever = db.as_retriever(search_kwargs={"k": 3})

llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# ---------- Request Model ----------

class QuestionRequest(BaseModel):
    question: str

# ---------- Upload Endpoint ----------

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    with open("temp.pdf", "wb") as f:
        f.write(await file.read())

    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    db.add_documents(chunks)

    return {"message": "Document uploaded successfully"}

# ---------- RAG Question Endpoint ----------

@app.post("/ask")
def ask_question(request: QuestionRequest):

    docs = retriever.invoke(request.question)

    context = ""
    for i, d in enumerate(docs):
        context += f"Source {i+1}:\n{d.page_content}\n\n"

    prompt = f"""
You are a document assistant.

Rules:
- Answer only from the provided sources
- If not found, say "Not in document"
- Include source numbers

Sources:
{context}

Question:
{request.question}

Answer:
"""

    response = llm.invoke(prompt)

    return {
        "question": request.question,
        "answer": response.content
    }

# uvicorn app:app --reload