
import hashlib
import json
from pathlib import Path

import fitz
import pytesseract
import streamlit as st
from PIL import Image


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

CACHE_DIR = Path(".minidoc_cache")

BATCH_SIZE = 25
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

OCR_ZOOM = 1.5

# Windows Tesseract path
TESSERACT_PATH = Path(
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

if TESSERACT_PATH.exists():
    pytesseract.pytesseract.tesseract_cmd = str(
        TESSERACT_PATH
    )


# --------------------------------------------------
# PDF INFORMATION
# --------------------------------------------------

def get_document_hash(pdf_bytes):
    """Generate a unique SHA256 hash for the PDF."""

    return hashlib.sha256(pdf_bytes).hexdigest()


def get_pdf_page_count(pdf_bytes):
    """Return the total number of pages in a PDF."""

    pdf = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    try:
        return len(pdf)
    finally:
        pdf.close()


# --------------------------------------------------
# OCR CACHE
# --------------------------------------------------

def get_document_cache_dir(document_hash):
    """Create and return the document's cache folder."""

    folder = CACHE_DIR / document_hash
    folder.mkdir(parents=True, exist_ok=True)

    return folder


def get_page_cache_path(document_hash, page_number):
    """Return the JSON cache path for a page."""

    folder = get_document_cache_dir(document_hash)

    return folder / f"page_{page_number:06d}.json"


def load_cached_page(document_hash, page_number):
    """Load a previously processed page from cache."""

    cache_file = get_page_cache_path(
        document_hash,
        page_number
    )

    if not cache_file.exists():
        return None

    try:
        with open(cache_file, "r", encoding="utf-8") as file:
            return json.load(file)

    except (json.JSONDecodeError, OSError):
        return None


def save_page_cache(
    document_hash,
    page_number,
    text,
    used_ocr,
    error=None
):
    """Save page results safely using an atomic file replace."""

    cache_file = get_page_cache_path(
        document_hash,
        page_number
    )

    temporary_file = cache_file.with_suffix(".tmp")

    data = {
        "page": page_number,
        "text": text,
        "used_ocr": used_ocr,
        "error": error,
    }

    with open(
        temporary_file,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(data, file, ensure_ascii=False)

    temporary_file.replace(cache_file)


# --------------------------------------------------
# TEXT EXTRACTION AND OCR
# --------------------------------------------------

def extract_page_text(page):
    """
    Extract normal PDF text first.
    If no text is available, use OCR.
    """

    text = page.get_text("text").strip()

    if text:
        return text, False

    # Render the page as an image for OCR
    matrix = fitz.Matrix(
        OCR_ZOOM,
        OCR_ZOOM
    )

    pixmap = page.get_pixmap(
        matrix=matrix,
        alpha=False
    )

    image = Image.frombytes(
        "RGB",
        [pixmap.width, pixmap.height],
        pixmap.samples
    )

    text = pytesseract.image_to_string(
        image,
        lang="eng"
    ).strip()

    return text, True


# --------------------------------------------------
# CHUNKING
# --------------------------------------------------

def create_chunks(
    pages,
    document_name,
    document_hash
):
    """
    Create text chunks from all cached pages.

    Each chunk stores its document name,
    document hash and page number.
    """

    chunks = []

    for page_data in pages:

        text = page_data.get("text", "").strip()

        if not text:
            continue

        words = text.split()

        if not words:
            continue

        start = 0

        while start < len(words):

            end = start + CHUNK_SIZE

            chunk_text = " ".join(
                words[start:end]
            )

            if chunk_text.strip():

                chunks.append({
                    "text": chunk_text,

                    "metadata": {
                        "document_name": document_name,
                        "document_hash": document_hash,
                        "page": page_data["page"],
                    }
                })

            if end >= len(words):
                break

            start = end - CHUNK_OVERLAP

    return chunks


# --------------------------------------------------
# DOCUMENT CACHE SUMMARY
# --------------------------------------------------

def get_cached_document_pages(
    document_hash,
    total_pages
):
    """Load every cached page for this document."""

    cached_pages = []

    for page_number in range(1, total_pages + 1):

        page_data = load_cached_page(
            document_hash,
            page_number
        )

        if page_data is not None:
            cached_pages.append(page_data)

    return cached_pages


# --------------------------------------------------
# BATCH PDF PROCESSING
# --------------------------------------------------

def process_pdf_batch(
    pdf_bytes,
    document_name,
    start_page=1,
    end_page=None,
    batch_size=BATCH_SIZE
):
    """
    Process a limited number of uncached pages.

    Previously cached pages are skipped.

    Returns chunks built from ALL cached pages,
    not just the current batch.
    """

    pdf_bytes = bytes(pdf_bytes)

    document_hash = get_document_hash(
        pdf_bytes
    )

    pdf = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    total_pages = len(pdf)

    if total_pages == 0:
        pdf.close()
        raise ValueError("The PDF contains no pages.")

    if end_page is None:
        end_page = total_pages

    start_page = max(1, int(start_page))
    end_page = min(total_pages, int(end_page))
    batch_size = max(1, int(batch_size))

    if start_page > end_page:
        pdf.close()
        raise ValueError("Invalid page range.")

    # Find uncached pages in the selected range
    uncached_pages = []

    for page_number in range(start_page, end_page + 1):

        cached = load_cached_page(
            document_hash,
            page_number
        )

        if cached is None:
            uncached_pages.append(page_number)

    # Select only the next batch
    pages_to_process = uncached_pages[:batch_size]

    batch_total = len(pages_to_process)
    batch_processed = 0

    progress_bar = st.progress(0)

    status_text = st.empty()

    try:

        for index, page_number in enumerate(
            pages_to_process,
            start=1
        ):

            status_text.info(
                f"Processing page {page_number} "
                f"/ {total_pages} "
                f"({index}/{batch_total} in this batch)"
            )

            page = pdf.load_page(
                page_number - 1
            )

            try:

                text, used_ocr = extract_page_text(
                    page
                )

                save_page_cache(
                    document_hash=document_hash,
                    page_number=page_number,
                    text=text,
                    used_ocr=used_ocr
                )

            except Exception as error:

                # Save the error so one problematic page
                # does not repeatedly stop the batch.
                save_page_cache(
                    document_hash=document_hash,
                    page_number=page_number,
                    text="",
                    used_ocr=False,
                    error=str(error)
                )

                st.warning(
                    f"Page {page_number} could not be processed: "
                    f"{error}"
                )

            batch_processed += 1

            progress_bar.progress(
                index / batch_total
            )

    finally:
        pdf.close()

    # Load ALL cached pages, including earlier batches
    all_cached_pages = get_cached_document_pages(
        document_hash,
        total_pages
    )

    # Rebuild chunks from all cached pages
    chunks = create_chunks(
        pages=all_cached_pages,
        document_name=document_name,
        document_hash=document_hash
    )

    processed_pages = len(all_cached_pages)

    ocr_pages = sum(
        1
        for page in all_cached_pages
        if page.get("used_ocr", False)
    )

    # Find the next uncached page in the selected range
    remaining_pages = []

    for page_number in range(start_page, end_page + 1):

        if load_cached_page(
            document_hash,
            page_number
        ) is None:

            remaining_pages.append(page_number)

    completed = len(remaining_pages) == 0

    next_page = (
        remaining_pages[0]
        if remaining_pages
        else end_page + 1
    )

    if completed:
        status_text.success(
            f"Selected pages completed! "
            f"{processed_pages}/{total_pages} pages cached."
        )
    else:
        status_text.success(
            f"Batch completed! Next page: {next_page}"
        )

    return {
        "document_name": document_name,
        "document_hash": document_hash,
        "total_pages": total_pages,
        "processed_pages": processed_pages,
        "ocr_pages": ocr_pages,
        "batch_processed": batch_processed,
        "next_page": next_page,
        "completed": completed,
        "chunks": chunks,
    }
