
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama
import io
import hashlib
import uuid

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="MiniDoc AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# 2. CUSTOM CSS - DARK SAAS DESIGN
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #0b0d14;
    --panel: #121622;
    --panel2: #191f2e;
    --border: #252d40;
    --text: #f2f4fa;
    --muted: #929bb0;
    --accent: #8b7cff;
    --accent2: #6654e8;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: var(--bg);
    color: var(--text);
}

[data-testid="stHeader"] {
    background: rgba(11,13,20,0.9);
}

[data-testid="stSidebar"] {
    background: #10131d;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.brand {
    font-size: 25px;
    font-weight: 800;
    letter-spacing: -1px;
    color: white;
    margin-bottom: 4px;
}

.brand span {
    color: var(--accent);
}

.tagline {
    color: var(--muted);
    font-size: 12px;
    margin-bottom: 25px;
}

.section-label {
    color: #858fa7;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-top: 22px;
    margin-bottom: 10px;
}

.hero {
    padding: 30px;
    border-radius: 18px;
    border: 1px solid var(--border);
    background: linear-gradient(135deg, #171b2a, #111521);
    margin-bottom: 25px;
}

.hero h1 {
    font-size: clamp(25px, 4vw, 38px);
    font-weight: 800;
    letter-spacing: -1.5px;
    margin: 0 0 10px 0;
    color: #ffffff;
}

.hero p {
    color: var(--muted);
    font-size: 14px;
    margin: 0;
}

.info-card {
    background: var(--panel);
    border: 1px solid var(--border);
    padding: 18px;
    border-radius: 14px;
    margin-bottom: 15px;
}

.info-card h4 {
    color: white;
    margin: 0 0 6px 0;
    font-size: 14px;
}

.info-card p {
    color: var(--muted);
    font-size: 12px;
    margin: 0;
}

.stButton > button {
    border-radius: 10px;
    border: 1px solid var(--border);
    background: var(--panel2);
    color: white;
    font-weight: 600;
    min-height: 42px;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    border-color: var(--accent);
    color: white;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #7565f5, #8d71ff);
    border: none;
    color: white;
}

.stTextInput input, .stTextArea textarea {
    background: var(--panel);
    color: white;
    border: 1px solid var(--border);
    border-radius: 10px;
}

[data-testid="stFileUploader"] {
    background: var(--panel);
    border: 1px dashed #454e68;
    border-radius: 12px;
    padding: 10px;
}

[data-testid="stChatMessage"] {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 15px;
    margin-bottom: 12px;
}

.stChatInput textarea {
    background: var(--panel);
    color: white;
}

[data-testid="stExpander"] {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
}

hr {
    border-color: var(--border);
}

.muted {
    color: var(--muted);
    font-size: 12px;
}

@media (max-width: 768px) {
    .block-container {
        padding: 1rem 0.8rem 2rem 0.8rem;
    }

    .hero {
        padding: 20px;
        border-radius: 14px;
    }

    .hero h1 {
        font-size: 26px;
    }

    [data-testid="stSidebar"] {
        min-width: 0;
    }

    [data-testid="stChatMessage"] {
        padding: 10px;
    }
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 3. SESSION STATE
# --------------------------------------------------

def create_chat():
    chat_id = uuid.uuid4().hex[:8]
    st.session_state.chats[chat_id] = {
        "title": "New conversation",
        "messages": []
    }
    st.session_state.active_chat = chat_id

if "chats" not in st.session_state:
    st.session_state.chats = {}
    create_chat()

if "active_chat" not in st.session_state:
    create_chat()

if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "pages" not in st.session_state:
    st.session_state.pages = []

if "uploaded_hash" not in st.session_state:
    st.session_state.uploaded_hash = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

# --------------------------------------------------
# 4. LOAD EMBEDDING MODEL
# --------------------------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# --------------------------------------------------
# 5. DOCUMENT CHUNKING
# --------------------------------------------------

def create_chunks(text, chunk_size=1000, overlap=150):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks

# --------------------------------------------------
# 6. RESET DOCUMENT WHEN A NEW PDF IS UPLOADED
# --------------------------------------------------

def reset_document():
    st.session_state.index = None
    st.session_state.chunks = []
    st.session_state.pages = []
    st.session_state.document_name = None
    st.session_state.chats = {}
    create_chat()

# --------------------------------------------------
# 7. SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown(
        '<div class="brand">MiniDoc<span>.AI</span></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="tagline">Your intelligent document assistant</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-label">Workspace</div>',
                unsafe_allow_html=True)

    if st.button("＋  New Chat", use_container_width=True):
        create_chat()
        st.rerun()

    st.markdown('<div class="section-label">Upload Document</div>',
                unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a PDF document",
        type=["pdf"],
        key="pdf_uploader"
    )

    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        current_hash = hashlib.sha256(file_bytes).hexdigest()

        if current_hash != st.session_state.uploaded_hash:
            reset_document()
            st.session_state.uploaded_hash = current_hash
            st.session_state.document_name = uploaded_file.name

        st.caption(f"📄 {uploaded_file.name}")

        if st.button(
            "⚡ Process PDF",
            type="primary",
            use_container_width=True
        ):
            try:
                with st.spinner("Reading and processing your PDF..."):

                    reader = PdfReader(io.BytesIO(file_bytes))

                    all_chunks = []
                    page_numbers = []

                    for page_num, page in enumerate(reader.pages):
                        text = page.extract_text() or ""
                        page_chunks = create_chunks(text)

                        all_chunks.extend(page_chunks)
                        page_numbers.extend(
                            [page_num + 1] * len(page_chunks)
                        )

                    if not all_chunks:
                        st.error("No readable text found in this PDF.")
                    else:
                        model = load_embedding_model()

                        embeddings = model.encode(
                            all_chunks,
                            normalize_embeddings=True,
                            show_progress_bar=False
                        )

                        index = faiss.IndexFlatIP(
                            embeddings.shape[1]
                        )

                        index.add(
                            np.asarray(embeddings, dtype="float32")
                        )

                        st.session_state.index = index
                        st.session_state.chunks = all_chunks
                        st.session_state.pages = page_numbers

                        st.success("PDF processed successfully!")

            except Exception as e:
                st.error(f"PDF processing error: {e}")

    if st.session_state.index is not None:
        st.markdown('<div class="section-label">Active Document</div>',
                    unsafe_allow_html=True)

        st.success("● Document ready")

        st.caption(st.session_state.document_name)

        st.caption(
            f"{len(st.session_state.chunks)} text chunks indexed"
        )

    st.markdown('<div class="section-label">Chat History</div>',
                unsafe_allow_html=True)

    for chat_id, chat in reversed(
        list(st.session_state.chats.items())
    ):
        label = "💬 " + chat["title"][:25]

        if st.button(
            label,
            key=f"chat_{chat_id}",
            use_container_width=True
        ):
            st.session_state.active_chat = chat_id
            st.rerun()

    st.divider()

    if st.button("🗑 Clear Current Chat", use_container_width=True):
        st.session_state.chats[st.session_state.active_chat][
            "messages"
        ] = []
        st.session_state.chats[st.session_state.active_chat][
            "title"
        ] = "New conversation"
        st.rerun()

    st.markdown(
        '<p class="muted">MiniDoc AI · Powered by RAG</p>',
        unsafe_allow_html=True
    )

# --------------------------------------------------
# 8. MAIN DASHBOARD
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <h1>Chat with your documents. ✨</h1>
    <p>
        Upload a PDF, ask questions, and get answers
        grounded in your document.
    </p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 9. FEATURE CARDS
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-card">
        <h4>📄 PDF Intelligence</h4>
        <p>Extract and understand your PDF documents.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h4>🔍 Smart Retrieval</h4>
        <p>Find relevant information using vector search.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card">
        <h4>🤖 AI Answers</h4>
        <p>Get answers generated using your document context.</p>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# 10. CHAT INTERFACE
# --------------------------------------------------

active_chat = st.session_state.chats[
    st.session_state.active_chat
]

st.subheader("💬 Document Assistant")

if st.session_state.index is None:

    st.markdown("""
    <div class="info-card">
        <h4>👋 Welcome to MiniDoc AI!</h4>
        <p>
            Start by uploading a PDF from the sidebar.
            Process it, then ask your first question.
        </p>
    </div>
    """, unsafe_allow_html=True)

else:

    for message in active_chat["messages"]:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("📚 View retrieved sources"):
                    for source in message["sources"]:
                        st.markdown(
                            f"**Page {source['page']}**"
                        )
                        st.write(source["text"])

    question = st.chat_input(
        "Ask anything about your PDF..."
    )

    if question:

        if not active_chat["messages"]:
            active_chat["title"] = question[:30]

        active_chat["messages"].append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        try:
            with st.chat_message("assistant"):

                with st.spinner("Thinking..."):

                    model = load_embedding_model()

                    query_embedding = model.encode(
                        [question],
                        normalize_embeddings=True
                    )

                    number_of_results = min(
                        3,
                        len(st.session_state.chunks)
                    )

                    scores, indices = st.session_state.index.search(
                        np.asarray(
                            query_embedding,
                            dtype="float32"
                        ),
                        number_of_results
                    )

                    retrieved_chunks = []

                    for idx in indices[0]:
                        if idx >= 0:
                            retrieved_chunks.append({
                                "page": st.session_state.pages[idx],
                                "text": st.session_state.chunks[idx]
                            })

                    context = "\n\n".join([
                        f"Page {item['page']}: {item['text']}"
                        for item in retrieved_chunks
                    ])

                    prompt = f"""
You are MiniDoc AI, a helpful PDF assistant.

Answer the user's question using ONLY the provided PDF context.

Instructions:
1. Read the context carefully.
2. Answer in simple, clear English.
3. Information may be expressed in different words.
4. Do not invent facts or use outside knowledge.
5. If the answer is genuinely missing, say:
   "I couldn't find this information in the PDF."

PDF CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

                    response = ollama.chat(
                        model="llama3.2:1b",
                        messages=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        options={
                            "temperature": 0.2,
                            "num_predict": 300
                        }
                    )

                    answer = response["message"]["content"]

                    st.markdown(answer)

                    with st.expander("📚 View retrieved sources"):
                        for item in retrieved_chunks:
                            st.markdown(
                                f"**Page {item['page']}**"
                            )
                            st.write(item["text"])

                    active_chat["messages"].append({
                        "role": "assistant",
                        "content": answer,
                        "sources": retrieved_chunks
                    })

        except Exception as e:

            error_message = (
                "I couldn't generate an answer. "
                "Please check that Ollama is running and "
                "llama3.2:1b is installed."
            )

            st.error(error_message)
            st.caption(str(e))

            active_chat["messages"].append({
                "role": "assistant",
                "content": error_message
            })

