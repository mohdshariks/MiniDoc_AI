
import uuid

import streamlit as st

st.info(
    "🚧 MiniDoc AI is currently under development. "
    "Some features may change or experience issues."
)

from config import setup_app, init_state
from pdf_processor import (
    process_pdf_batch,
    get_document_hash,
    get_pdf_page_count,
    BATCH_SIZE,
)
from retrieval import (
    build_subject_index,
    retrieve_chunks,
)
from llm import generate_answer
from ui import (
    show_pdf_viewer,
    render_sources,
)


# ==================================================
# 1. APP INITIALIZATION
# ==================================================

setup_app()
init_state()


# ==================================================
# 2. SUBJECT MANAGEMENT
# ==================================================

def create_subject(name):
    """Create a new subject."""

    subject_id = uuid.uuid4().hex[:8]

    st.session_state.subjects[subject_id] = {
        "id": subject_id,
        "name": name,
        "documents": {},
        "chunks": [],
        "faiss_index": None,
        "bm25_index": None,
        "ready": False,
        "messages": [],
    }

    st.session_state.active_subject = subject_id


def get_active_subject():
    """Return the currently selected subject."""

    subject_id = st.session_state.active_subject

    if subject_id is None:
        return None

    return st.session_state.subjects.get(subject_id)


# Create a default subject on first launch
if not st.session_state.subjects:
    create_subject("My First Subject")


# ==================================================
# 3. SIDEBAR
# ==================================================

with st.sidebar:

    st.markdown("## 📚 MiniDoc AI")

    st.caption("Your personal PDF study assistant")

    st.divider()

    # ----------------------------------------------
    # Subject selection
    # ----------------------------------------------

    st.markdown("### 📖 My Subjects")

    subject_ids = list(
        st.session_state.subjects.keys()
    )

    subject_names = {
        subject_id: st.session_state.subjects[subject_id]["name"]
        for subject_id in subject_ids
    }

    current_subject_id = st.session_state.active_subject

    if current_subject_id not in subject_ids:
        current_subject_id = subject_ids[0]

    selected_subject = st.selectbox(
        "Select subject",
        options=subject_ids,
        index=subject_ids.index(current_subject_id),
        format_func=lambda subject_id: subject_names[subject_id],
        key="subject_selector",
    )

    if selected_subject != st.session_state.active_subject:

        st.session_state.active_subject = selected_subject

        st.session_state.viewer_document = None
        st.session_state.viewer_page = 1

        st.rerun()

    # ----------------------------------------------
    # Create subject
    # ----------------------------------------------

    with st.expander("➕ Create New Subject"):

        new_subject_name = st.text_input(
            "Subject name",
            placeholder="Example: Computer Science",
            key="new_subject_name",
        )

        if st.button(
            "Create Subject",
            use_container_width=True,
        ):

            if new_subject_name.strip():

                create_subject(
                    new_subject_name.strip()
                )

                st.session_state.new_subject_name = ""

                st.rerun()

            else:
                st.warning("Please enter a subject name.")

    subject = get_active_subject()

    st.divider()

    # ----------------------------------------------
    # PDF uploader
    # ----------------------------------------------

    st.markdown("### 📤 Upload PDFs")

    uploaded_files = st.file_uploader(
        "Upload your study materials",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
    )

    st.caption(
        f"Maximum {BATCH_SIZE} uncached pages per PDF "
        "per processing click."
    )

    # ----------------------------------------------
    # Process PDFs
    # ----------------------------------------------

    if st.button(
        "⚙️ Process PDFs",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files,
    ):

        for uploaded_file in uploaded_files:

            pdf_bytes = uploaded_file.getvalue()

            document_name = uploaded_file.name

            document_hash = get_document_hash(
                pdf_bytes
            )

            st.markdown("---")

            st.markdown(
                f"**📄 Processing: {document_name}**"
            )

            try:

                total_pages = get_pdf_page_count(
                    pdf_bytes
                )

                # Create or retrieve document record
                if document_hash not in subject["documents"]:

                    subject["documents"][document_hash] = {
                        "name": document_name,
                        "hash": document_hash,
                        "bytes": pdf_bytes,
                        "chunks": [],
                        "total_pages": total_pages,
                        "processed_pages": 0,
                        "ocr_pages": 0,
                        "next_page": 1,
                        "completed": False,
                    }

                document = subject["documents"][document_hash]

                # Refresh PDF bytes and name
                document["name"] = document_name
                document["bytes"] = pdf_bytes
                document["total_pages"] = total_pages

                start_page = document.get(
                    "next_page",
                    1
                )

                # If this document was already processed,
                # rebuild its chunks from the persistent cache.
                if start_page > total_pages:

                    start_page = 1
                    end_page = total_pages

                else:

                    end_page = total_pages

                result = process_pdf_batch(
                    pdf_bytes=pdf_bytes,
                    document_name=document_name,
                    start_page=start_page,
                    end_page=end_page,
                    batch_size=BATCH_SIZE,
                )

                # Update document information
                document["chunks"] = result["chunks"]

                document["processed_pages"] = result["processed_pages"]

                document["ocr_pages"] = result["ocr_pages"]

                document["next_page"] = result["next_page"]

                document["completed"] = result["completed"]

                # Build the subject's complete search index
                indexed = build_subject_index(
                    subject
                )

                if result["completed"]:

                    st.success(
                        f"Completed: {document_name}"
                    )

                else:

                    st.info(
                        f"Batch completed for {document_name}. "
                        f"Next page: {result['next_page']}"
                    )

                st.write(
                    f"Pages cached: {result['processed_pages']} "
                    f"/ {result['total_pages']}"
                )

                st.write(
                    f"OCR pages: {result['ocr_pages']}"
                )

                st.write(
                    f"Chunks created: {len(result['chunks'])}"
                )

                if not indexed:

                    st.warning(
                        "No searchable text was found in this PDF."
                    )

            except Exception as error:

                st.error(
                    f"Could not process {document_name}: {error}"
                )

        st.rerun()

    # ----------------------------------------------
    # Processing status
    # ----------------------------------------------

    st.divider()

    st.markdown("### 📊 Document Status")

    if subject["documents"]:

        for document in subject["documents"].values():

            with st.expander(
                f"📄 {document['name']}"
            ):

                st.write(
                    f"Total pages: {document['total_pages']}"
                )

                st.write(
                    f"Cached pages: {document['processed_pages']}"
                )

                st.write(
                    f"OCR pages: {document['ocr_pages']}"
                )

                st.write(
                    f"Next page: {document['next_page']}"
                )

                if document["completed"]:

                    st.success("Processing completed")

                else:

                    st.warning("More pages need processing")

    else:

        st.caption("No PDFs uploaded yet.")

    # ----------------------------------------------
    # Clear PDFs
    # ----------------------------------------------

    st.divider()

    if st.button(
        "🗑️ Clear Subject PDFs",
        use_container_width=True,
    ):

        subject["documents"] = {}
        subject["chunks"] = []
        subject["faiss_index"] = None
        subject["bm25_index"] = None
        subject["ready"] = False

        st.session_state.viewer_document = None
        st.session_state.viewer_page = 1

        st.rerun()

    # ----------------------------------------------
    # New conversation
    # ----------------------------------------------

    if st.button(
        "➕ New Conversation",
        use_container_width=True,
    ):

        subject["messages"] = []

        st.rerun()

    # ----------------------------------------------
    # Theme preference
    # ----------------------------------------------

    st.divider()

    st.markdown("### ⚙️ Preferences")

    st.selectbox(
        "Theme preference",
        ["Dark", "Light"],
        key="theme",
    )

    st.caption(
        "The theme preference is saved for future UI customization."
    )


# ==================================================
# 4. MAIN APPLICATION
# ==================================================

subject = get_active_subject()

if subject is None:

    st.warning("Please select or create a subject.")

    st.stop()


# ----------------------------------------------
# Header
# ----------------------------------------------

st.markdown(
    '<div class="main-title">📚 MiniDoc AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    f'<div class="subtitle">'
    f'Your AI study assistant · {subject["name"]}'
    f'</div>',
    unsafe_allow_html=True,
)


# ==================================================
# 5. CHECK DOCUMENT AVAILABILITY
# ==================================================

if not subject["documents"]:

    st.info(
        "👋 Welcome to MiniDoc AI! "
        "Upload your study PDFs from the sidebar "
        "and click Process PDFs to get started."
    )

    st.stop()


if not subject["ready"]:

    st.warning(
        "Your PDFs have not been indexed yet. "
        "Please process your PDFs from the sidebar."
    )

    st.stop()


# ==================================================
# 6. CHAT AND PDF VIEWER LAYOUT
# ==================================================

chat_column, viewer_column = st.columns(
    [1.05, 1],
    gap="large",
)


# ==================================================
# 7. CHAT INTERFACE
# ==================================================

with chat_column:

    st.markdown("### 💬 Ask Your PDFs")

    st.caption(
        "Ask questions about your uploaded study materials."
    )

    # ----------------------------------------------
    # Welcome message
    # ----------------------------------------------

    if not subject["messages"]:

        with st.chat_message("assistant"):

            st.markdown(
                """
                👋 Hello! Welcome to MiniDoc AI.

                I can help you understand your uploaded PDFs.

                **You can ask me:**
                - Explain a topic in simple English.
                - Summarize a chapter.
                - Answer questions from your study material.
                - Find definitions and important points.
                - Explain multiple-choice questions.
                """
            )

    # ----------------------------------------------
    # Display chat history
    # ----------------------------------------------

    for message_index, message in enumerate(
        subject["messages"]
    ):

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

            if message["role"] == "assistant":

                sources = message.get(
                    "sources",
                    []
                )

    # ----------------------------------------------
    # Chat input
    # ----------------------------------------------

    question = st.chat_input(
        "Ask a question about your PDFs..."
    )

    if question:

        # Save student question
        subject["messages"].append({
            "role": "user",
            "content": question,
        })

        # Display question
        with st.chat_message("user"):

            st.markdown(question)

        # Retrieve relevant PDF chunks
        with st.chat_message("assistant"):

            with st.spinner(
                "Searching your PDFs..."
            ):

                retrieved_chunks = retrieve_chunks(
                    query=question,
                    subject=subject,
                    top_k=12,
                )

            if retrieved_chunks:

                with st.spinner(
                    "Preparing your answer..."
                ):

                    answer = generate_answer(
                        question=question,
                        retrieved_chunks=retrieved_chunks,
                    )

            else:

                answer = (
                    "I couldn't find relevant information "
                    "in your uploaded PDFs. "
                    "Try asking the question differently."
                )

            st.markdown(answer)

        # Save assistant response
        subject["messages"].append({
            "role": "assistant",
            "content": answer,
            "sources": retrieved_chunks,
        })

        st.rerun()


# ==================================================
# 8. PDF VIEWER
# ==================================================

with viewer_column:

    st.markdown("### 📖 PDF Reader")

    document_hash = st.session_state.viewer_document

    # Select first document by default
    if (
        document_hash not in subject["documents"]
        or document_hash is None
    ):

        document_hash = next(
            iter(subject["documents"])
        )

        st.session_state.viewer_document = document_hash

    selected_document = subject["documents"][
        document_hash
    ]

    document_options = list(
        subject["documents"].keys()
    )

    selected_document_hash = st.selectbox(
        "Choose PDF",
        options=document_options,
        index=document_options.index(document_hash),
        format_func=lambda key: subject["documents"][key]["name"],
        key="viewer_document_selector",
    )

    if selected_document_hash != document_hash:

        st.session_state.viewer_document = selected_document_hash

        st.session_state.viewer_page = 1

        st.rerun()

    selected_document = subject["documents"][
        selected_document_hash
    ]

    total_pages = selected_document["total_pages"]

    # Page selection
    selected_page = st.number_input(
        "Selected source page",
        min_value=1,
        max_value=max(1, total_pages),
        value=min(
            max(1, st.session_state.viewer_page),
            max(1, total_pages),
        ),
        step=1,
        key="viewer_page_selector",
    )

    st.session_state.viewer_page = selected_page

    # Display PDF
    show_pdf_viewer(
        document=selected_document,
        page_number=selected_page,
    )
