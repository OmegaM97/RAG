import streamlit as st
import requests

BACKEND_URL = "http://127.0.0.1:8000"

if "history" not in st.session_state:
    st.session_state.history = []

st.title("📚 RAG Assistant")


uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    files = {"file": (uploaded_file.name,
                      uploaded_file.getvalue(), 
                      "application/pdf")}
    
    response = requests.post(f"{BACKEND_URL}/upload", files=files)
    
    if response.status_code == 200:
        st.success(response.json()["message"])
    else:
        st.error(response.text)
question = st.text_input("Ask a question")

if st.button("Ask") and question:  
    response = requests.post(
        f"{BACKEND_URL}/ask",
        json={"question": question}
    )
    
    answer = response.json()["answer"]
    
   
    st.write("### Answer")
    st.write(answer)
    st.session_state.history.insert(0, {
        "question": question,
        "answer": answer
    })
    
    st.session_state.history = st.session_state.history[:10]

st.divider()
st.subheader(" Search History (Last 10)")

for item in st.session_state.history:
    with st.expander(item["question"]):
        st.write(item["answer"])