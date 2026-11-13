
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama
import io

st.set_page_config(page_title="MiniDoc AI")
if st.button("🔄 Reset / Upload New PDF"):
    for key in ["index", "chunks", "pages", "chat_history"]:
        st.session_state.pop(key, None)

    st.rerun()
    
st.title("MiniDoc AI - PDF Chatbot")

# Load embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_embedding_model()

# Split text into chunks
def create_chunks(text, chunk_size=800, overlap=120):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# Upload PDF
uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

if uploaded_file:
    reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))

    all_chunks = []
    page_numbers = []

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text() or ""

        chunks = create_chunks(text)

        all_chunks.extend(chunks)
        page_numbers.extend([page_num + 1] * len(chunks))

    if not all_chunks:
        st.error("No readable text found in this PDF.")
        st.stop()

    # Create embeddings
    if st.button("Process PDF"):
        embeddings = model.encode(
            all_chunks,
            normalize_embeddings=True
        )

        # Create FAISS index
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(np.array(embeddings, dtype="float32"))

        st.session_state.index = index
        st.session_state.chunks = all_chunks
        st.session_state.pages = page_numbers

        st.success("PDF processed successfully!")

# Ask questions
if "index" in st.session_state:

    question = st.text_input("Ask a question about your PDF")

    if st.button("Get Answer") and question:

        # Convert question into embedding
        query_embedding = model.encode(
            [question],
            normalize_embeddings=True
        )

        # Retrieve relevant chunks
        scores, indices = st.session_state.index.search(
            np.array(query_embedding, dtype="float32"),
            3
        )

        retrieved_chunks = []

        for idx in indices[0]:
            if idx >= 0:
                chunk = st.session_state.chunks[idx]
                page = st.session_state.pages[idx]

                retrieved_chunks.append(
                    f"Page {page}: {chunk}"
                )

        context = "\n\n".join(retrieved_chunks)

        prompt = f"""
        You are a helpful PDF assistant.

        Answer the user's question using the information
        provided in the context below.

        Instructions:
        1. Read the entire context carefully.
        2. Find information related to the question.
        3. Answer in simple, clear English.
        4. Do not reject an answer if the information is
        available in different words.
        5. Do not make up information.
        6. If the answer is genuinely missing, say:
        "I couldn't find this information in the PDF."

        PDF CONTEXT:
        {context}

        USER QUESTION:
        {question}

        ANSWER:
        """

        # Generate answer using Ollama
        response = ollama.chat(
            model="llama3.2:1b",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        st.subheader("AI Answer")
        st.write(response["message"]["content"])

        with st.expander("View retrieved sources"):
            for chunk in retrieved_chunks:
                st.write(chunk)
