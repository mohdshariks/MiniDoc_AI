
import re

import faiss
import numpy as np
import streamlit as st

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

EMBEDDING_BATCH_SIZE = 64

VECTOR_SEARCH_LIMIT = 20
KEYWORD_SEARCH_LIMIT = 20

DEFAULT_TOP_K = 12

RRF_CONSTANT = 60


# --------------------------------------------------
# EMBEDDING MODEL
# --------------------------------------------------

@st.cache_resource(show_spinner="Loading AI embedding model...")
def load_embedding_model():
    """
    Load the SentenceTransformer model once.
    Streamlit caches it to avoid repeated loading.
    """

    return SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )


# --------------------------------------------------
# TEXT TOKENIZATION
# --------------------------------------------------

def tokenize(text):
    """
    Convert text into lowercase searchable words.
    """

    return re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower()
    )


# --------------------------------------------------
# CREATE EMBEDDINGS
# --------------------------------------------------

def create_embeddings(texts):
    """
    Convert text chunks into numerical vectors.
    """

    if not texts:
        return np.empty(
            (0, 384),
            dtype="float32"
        )

    model = load_embedding_model()

    embeddings = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return np.asarray(
        embeddings,
        dtype="float32"
    )


# --------------------------------------------------
# BUILD SUBJECT INDEX
# --------------------------------------------------

def build_subject_index(subject):
    """
    Build FAISS and BM25 indexes for all subject PDFs.

    This function should be called after PDF processing
    or whenever the subject's document collection changes.
    """

    all_chunks = []

    # Collect chunks from every document
    for document in subject["documents"].values():

        document_chunks = document.get(
            "chunks",
            []
        )

        all_chunks.extend(
            document_chunks
        )

    if not all_chunks:

        subject["chunks"] = []
        subject["faiss_index"] = None
        subject["bm25_index"] = None
        subject["ready"] = False

        return False

    texts = [
        chunk["text"]
        for chunk in all_chunks
        if chunk.get("text", "").strip()
    ]

    if not texts:

        subject["chunks"] = []
        subject["faiss_index"] = None
        subject["bm25_index"] = None
        subject["ready"] = False

        return False

    # Keep chunk metadata aligned with the text list
    valid_chunks = [
        chunk
        for chunk in all_chunks
        if chunk.get("text", "").strip()
    ]

    with st.spinner(
        "Building your PDF search index..."
    ):

        # Create vector embeddings
        embeddings = create_embeddings(
            texts
        )

        # Create FAISS index
        dimension = embeddings.shape[1]

        faiss_index = faiss.IndexFlatIP(
            dimension
        )

        faiss_index.add(
            embeddings
        )

        # Create BM25 keyword index
        tokenized_texts = [
            tokenize(text)
            for text in texts
        ]

        bm25_index = BM25Okapi(
            tokenized_texts
        )

    # Save indexes in subject data
    subject["chunks"] = valid_chunks

    subject["faiss_index"] = faiss_index

    subject["bm25_index"] = bm25_index

    subject["ready"] = True

    return True


# --------------------------------------------------
# VECTOR SEARCH
# --------------------------------------------------

def vector_search(
    query,
    subject,
    limit=VECTOR_SEARCH_LIMIT
):
    """
    Find relevant chunks using FAISS.
    """

    index = subject.get(
        "faiss_index"
    )

    chunks = subject.get(
        "chunks",
        []
    )

    if index is None or not chunks:
        return []

    query_embedding = create_embeddings(
        [query]
    )

    scores, indices = index.search(
        query_embedding,
        min(limit, len(chunks))
    )

    results = []

    for score, index_position in zip(
        scores[0],
        indices[0]
    ):

        if index_position < 0:
            continue

        results.append({
            "chunk": chunks[index_position],
            "score": float(score)
        })

    return results


# --------------------------------------------------
# KEYWORD SEARCH
# --------------------------------------------------

def keyword_search(
    query,
    subject,
    limit=KEYWORD_SEARCH_LIMIT
):
    """
    Find relevant chunks using BM25.
    """

    bm25_index = subject.get(
        "bm25_index"
    )

    chunks = subject.get(
        "chunks",
        []
    )

    if bm25_index is None or not chunks:
        return []

    query_tokens = tokenize(
        query
    )

    if not query_tokens:
        return []

    scores = bm25_index.get_scores(
        query_tokens
    )

    ranked_indices = np.argsort(
        scores
    )[::-1][:limit]

    results = []

    for index_position in ranked_indices:

        score = float(
            scores[index_position]
        )

        if score <= 0:
            continue

        results.append({
            "chunk": chunks[index_position],
            "score": score
        })

    return results


# --------------------------------------------------
# RECIPROCAL RANK FUSION
# --------------------------------------------------

def reciprocal_rank_fusion(
    vector_results,
    keyword_results
):
    """
    Combine FAISS and BM25 rankings using RRF.

    RRF rewards chunks that rank highly in both searches.
    """

    combined_scores = {}

    chunk_lookup = {}

    def get_chunk_key(chunk):

        metadata = chunk.get(
            "metadata",
            {}
        )

        return (
            metadata.get("document_hash", ""),
            metadata.get("page", 0),
            chunk.get("text", "")
        )

    # Process vector search results
    for rank, result in enumerate(
        vector_results,
        start=1
    ):

        chunk = result["chunk"]

        key = get_chunk_key(
            chunk
        )

        combined_scores[key] = (
            combined_scores.get(key, 0)
            + 1 / (RRF_CONSTANT + rank)
        )

        chunk_lookup[key] = chunk

    # Process keyword search results
    for rank, result in enumerate(
        keyword_results,
        start=1
    ):

        chunk = result["chunk"]

        key = get_chunk_key(
            chunk
        )

        combined_scores[key] = (
            combined_scores.get(key, 0)
            + 1 / (RRF_CONSTANT + rank)
        )

        chunk_lookup[key] = chunk

    # Sort by combined score
    ranked_keys = sorted(
        combined_scores,
        key=lambda key: combined_scores[key],
        reverse=True
    )

    results = []

    for key in ranked_keys:

        results.append({
            "chunk": chunk_lookup[key],
            "score": combined_scores[key]
        })

    return results


# --------------------------------------------------
# HYBRID RETRIEVAL
# --------------------------------------------------

def retrieve_chunks(
    query,
    subject,
    top_k=DEFAULT_TOP_K
):
    """
    Main retrieval function.

    1. Search with FAISS.
    2. Search with BM25.
    3. Combine results using RRF.
    4. Return the top matching chunks.
    """

    if not subject.get("ready", False):
        return []

    if not query.strip():
        return []

    # FAISS vector search
    vector_results = vector_search(
        query=query,
        subject=subject
    )

    # BM25 keyword search
    keyword_results = keyword_search(
        query=query,
        subject=subject
    )

    # Combine both rankings
    combined_results = reciprocal_rank_fusion(
        vector_results=vector_results,
        keyword_results=keyword_results
    )

    # Return only the best matches
    return [
        result["chunk"]
        for result in combined_results[:top_k]
    ]
