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

# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector database
db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# LLM
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY")
)


class QuestionRequest(BaseModel):
    question: str


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

@app.post("/ask")
def ask_question(request: QuestionRequest):

    docs = db.similarity_search(request.question, k=3)

    if not docs:
        return {"question": request.question, "answer": "No relevant documents found in the database."}

    context_text = "\n\n---\n\n".join(
        [f"Source {i+1}:\n{d.page_content}" for i, d in enumerate(docs)])

    system_prompt = (
        "You are a document assistant. Answer the user's question using ONLY the provided sources. "
        "If the answer is not in the sources, say 'Not in document'. "
        "Always cite the source numbers used in your answer."
    )

    user_prompt = f"Context:\n{context_text}\n\nQuestion: {request.question}"

    from langchain_core.messages import SystemMessage, HumanMessage

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    response = llm.invoke(messages)

    return {
        "question": request.question,
        "answer": response.content
    }