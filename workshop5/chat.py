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

# ---------- Initialize Embeddings & DB ----------

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

class Question(BaseModel):
    question: str

# ---------- Upload PDF ----------

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    with open("uploaded.pdf", "wb") as f:
        f.write(await file.read())

    loader = PyPDFLoader("uploaded.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    db.add_documents(chunks)

    return {"message": "Document uploaded and stored!"}

# ---------- Ask Question (RAG) ----------

@app.post("/ask")
def ask_question(data: Question):

    docs = retriever.invoke(data.question)

    context = ""
    for i, doc in enumerate(docs):
        context += f"Source {i+1}:\n{doc.page_content}\n\n"

    prompt = f"""
Answer using only the sources below.
If not found, say 'Not in document'.

{context}

Question: {data.question}
"""

    response = llm.invoke(prompt)

    return {
        "question": data.question,
        "answer": response.content
    }

# streamlit run chat.py