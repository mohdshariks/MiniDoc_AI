
import os
import streamlit as st


def setup_app():
    st.set_page_config(
        page_title="MiniDoc AI | Student Study Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)


def init_state():
    defaults = {
        "subjects": {},
        "active_subject": None,
        "viewer_document": None,
        "viewer_page": 1,
        "theme": "Dark",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_groq_api_key():
    try:
        key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        key = None

    return key or os.getenv("GROQ_API_KEY")


CSS = """
<style>
.stApp {
    background: var(--background-color);
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1600px;
}

[data-testid="stSidebar"] {
    border-right: 1px solid rgba(128,128,128,.2);
}

.main-title {
    font-size: 1.8rem;
    font-weight: 750;
    letter-spacing: -.7px;
    margin-bottom: 0;
}

.subtitle {
    color: #9299a5;
    font-size: .9rem;
    margin: 4px 0 20px;
}

[data-testid="stChatMessage"] {
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 12px;
}

[data-testid="stChatInput"] {
    border-radius: 18px !important;
}

[data-testid="stChatInput"] textarea {
    font-size: 15px !important;
}

.stButton button {
    border-radius: 9px;
}

iframe {
    border: 1px solid rgba(128,128,128,.25);
    border-radius: 12px;
}

@media (max-width: 900px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .main-title {
        font-size: 1.5rem;
    }
}
</style>
"""
