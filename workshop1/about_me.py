import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# -----------------------------
# Step 1: Load Environment Variables
# -----------------------------
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

# -----------------------------
# Step 2: Simulate Documents in File
# -----------------------------
documents = [
    {
        "title": "About me",
        "content": "I am omega melese, Addis ababa university student, im studying information science. im interested learning about AI"
    },
    {
        "title": "Life of Omega",
        "content": "he was born in 2005 in woldia, he is the last child of his family"
    }
]

# -----------------------------
# Step 3: Simple Retriever
# -----------------------------
def retrieve_doc(query):
    for doc in documents:
        if query.lower() in doc["title"].lower():
            return doc["content"]
    return "No document found."

# -----------------------------
# Step 4: Initialize Groq LLM
# -----------------------------
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0,
    groq_api_key=groq_api_key
)

# -----------------------------
# Step 5: Prompt Engineering Demo
# -----------------------------
print("=== Prompt Engineering Demo ===\n")

basic_question = "Explain Artificial Intelligence."
print("Basic Prompt Output:\n")
print(llm.invoke(basic_question).content)


improved_prompt = """
Explain Artificial Intelligence to a high school student.
Use simple language and 2 real-life examples.
"""

print("\nImproved Prompt Output:\n")
print(llm.invoke(improved_prompt).content)

# -----------------------------
# Step 6: RAG Demo
# -----------------------------
print("\n=== RAG Demo: About me ===\n")


# WITHOUT retrieval
print("Answer WITHOUT retrieval:\n")

user_question = "who am i"

print(llm.invoke(user_question).content)

# WITH retrieval
context = retrieve_doc("About me")

rag_prompt = f"""
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{user_question}
"""

print("\nAnswer WITH retrieval (RAG):\n")
print(llm.invoke(rag_prompt).content)

# -----------------------------
# Step 7: Workflow Visualization
# -----------------------------
print("\n=== RAG Workflow ===\n")
print("""
User Question
      ↓
Retriever (search documents list)
      ↓
Retrieved Context
      ↓
Context + Question → Prompt
      ↓
Groq LLM
      ↓
Final Answer
""")