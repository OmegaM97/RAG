from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# -----------------------------------------
# 1️⃣ Sample Documents
# -----------------------------------------

documents = [
    "My name is omega melese a 3rd year information science student",
    "i have been working on this internship called XXXX as backend developer",
    "i got 3rd place in a hackaton and i got gold medal in XXXX",
    "i always dreamed of becming a senior software developer",
    "now im working on this RAG project"
]

docs = [Document(page_content=text) for text in documents]

# -----------------------------------------
# 2️⃣ Load Embedding Model
# -----------------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------------------
# 3️⃣ Create Vector Store
# -----------------------------------------

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)

print("Vector database created successfully.")

# -----------------------------------------
# 4️⃣ Similarity Search Function
# -----------------------------------------

def search_query(query, top_k=2):
    print(f"\n🔎 Query: {query}")
    print(f"Top-K: {top_k}")

    results = vectorstore.similarity_search(query, k=top_k)

    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(result.page_content)

# -----------------------------------------
# 5️⃣ Run Example Searches
# -----------------------------------------

search_query("About me", top_k=2)
search_query("What is my goal", top_k=3)