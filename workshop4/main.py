import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_groq import ChatGroq

load_dotenv()

# 1 Load PDF
loader = PyPDFLoader("data/sample.pdf")
docs = loader.load()

# 2 Split text into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)

# 3 Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4 Store in vector database
db = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="./chroma_db"
)

retriever = db.as_retriever(search_kwargs={"k":3})

# 5 LLM
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

print("\nRAG System Ready\n")

while True:
    query = input("Ask: ")

    docs = retriever.invoke(query)

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You are a Resume Assistant AI. Your ONLY job is to answer questions strictly using the provided resume context.

RULES (MUST FOLLOW):

Source Restriction

Answer ONLY from the provided context.

Do NOT use prior knowledge.

Do NOT make assumptions.

Do NOT infer missing information.

Out-of-Document Handling

If the answer is not explicitly found in the context, respond EXACTLY with:
"You are out of document."

Do not add anything else.

No Hallucinations

Never fabricate skills, experience, projects, education, or achievements.

If something is partially mentioned but unclear, treat it as missing.

Resume-Focused Answers

Only answer questions about the person in the resume.

If the user asks general knowledge (e.g., "What is HTML?"), respond:
"You are out of document."

Even if a skill (like HTML) appears in the resume, only describe it in relation to the candidate, not explain the technology itself.

Response Format

Always respond in concise bullet points.

Use professional tone.

Do not include long paragraphs.

Conflict Handling

If multiple context chunks conflict, prefer the most specific and recent information.

Safety Check Before Answering
Before answering, silently verify:

Is the answer explicitly in the context?

Is the question about the resume subject?
If ANY answer is no → output:
"You are out of document."

OUTPUT STYLE:

Good Example:
User: What skills does he have?
Answer:

JavaScript

HTML

CSS

Bad Example (DO NOT DO):

Explaining what HTML is

Adding extra skills

Guessing experience

Remember: When unsure → "You are out of document."

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)

    print("\nAnswer:\n", response.content)