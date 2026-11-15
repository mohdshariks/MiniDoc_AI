
import streamlit as st


# --------------------------------------------------
# PDF VIEWER
# --------------------------------------------------

def show_pdf_viewer(document, page_number=1):
    """
    Display the selected PDF in the viewer.
    """

    st.markdown("### 📖 PDF Viewer")

    if document is None:
        st.info(
            "Select a PDF source from the chat "
            "to view the document here."
        )
        return

    pdf_bytes = document.get("bytes")

    document_name = document.get(
        "name",
        "Unknown PDF"
    )

    if not pdf_bytes:
        st.warning("PDF file data is not available.")
        return

    st.caption(f"📄 {document_name}")

    try:
        # Streamlit's PDF viewer displays the document.
        # The page number is shown separately because
        # st.pdf does not provide a page-jump argument.
        st.caption(
            f"Selected source page: {page_number}"
        )

        st.pdf(
            pdf_bytes,
            height=780,
            key=f"pdf_viewer_{document.get('hash', document_name)}"
        )

    except Exception as error:
        st.error(
            f"Could not display this PDF: {error}"
        )

        st.info(
            "Make sure your Streamlit version supports "
            "the PDF viewer."
        )


# --------------------------------------------------
# SOURCE INFORMATION
# --------------------------------------------------

def show_source_information(source, index=None):
    """
    Display information about one retrieved PDF source.
    """

    metadata = source.get(
        "metadata",
        {}
    )

    document_name = metadata.get(
        "document_name",
        "Unknown document"
    )

    page_number = metadata.get(
        "page",
        "Unknown"
    )

    heading = (
        f"Source {index}: {document_name}"
        if index is not None
        else document_name
    )

    with st.expander(
        heading,
        expanded=False
    ):

        st.markdown(
            f"**📄 PDF:** {document_name}"
        )

        st.markdown(
            f"**📍 Page:** {page_number}"
        )

        st.markdown("**Relevant content:**")

        st.write(
            source.get("text", "")
        )


# --------------------------------------------------
# SOURCE BUTTONS
# --------------------------------------------------

def render_sources(
    sources,
    documents,
    key_prefix="source"
):
    """
    Display source buttons below an AI response.

    Clicking a button selects the corresponding
    document and page in the PDF viewer.
    """

    if not sources:
        return

    st.markdown("**📚 Sources from your PDFs**")

    # Create a lookup for uploaded documents
    document_lookup = {}

    for document in documents.values():

        document_hash = document.get(
            "hash"
        )

        if document_hash:
            document_lookup[document_hash] = document

    for index, source in enumerate(
        sources,
        start=1
    ):

        metadata = source.get(
            "metadata",
            {}
        )

        document_hash = metadata.get(
            "document_hash",
            ""
        )

        document_name = metadata.get(
            "document_name",
            "Unknown PDF"
        )

        page_number = metadata.get(
            "page",
            1
        )

        document = document_lookup.get(
            document_hash
        )

        if document is None:
            st.caption(
                f"Source {index}: {document_name} "
                f"(PDF is not available)"
            )
            continue

        button_label = (
            f"📄 {document_name}  |  Page {page_number}"
        )

        button_key = (
            f"{key_prefix}_{index}_"
            f"{document_hash}_{page_number}"
        )

        if st.button(
            button_label,
            key=button_key,
            use_container_width=True
        ):

            st.session_state.viewer_document = document_hash

            st.session_state.viewer_page = int(
                page_number
            )

            st.rerun()


# --------------------------------------------------
# SELECTED PDF DETAILS
# --------------------------------------------------

def show_selected_source_details(
    document,
    page_number
):
    """
    Display the currently selected PDF and page details.
    """

    if document is None:
        return

    st.markdown("### Selected source")

    st.write(
        f"**PDF:** {document.get('name', 'Unknown PDF')}"
    )

    st.write(
        f"**Page:** {page_number}"
    )


# --------------------------------------------------
# CHAT SOURCE HISTORY
# --------------------------------------------------

def render_message_sources(
    messages,
    documents,
    key_prefix="history"
):
    """
    Display source information for chat history.

    Each assistant message can contain its own sources.
    """

    for message_index, message in enumerate(messages):

        if message.get("role") != "assistant":
            continue

        sources = message.get(
            "sources",
            []
        )

        if not sources:
            continue

        with st.expander(
            f"📚 Sources for answer {message_index + 1}"
        ):

            render_sources(
                sources=sources,
                documents=documents,
                key_prefix=(
                    f"{key_prefix}_{message_index}"
                )
            )
