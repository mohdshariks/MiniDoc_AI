# MiniDoc AI – PDF Question Answering Chatbot

MiniDoc AI is a beginner-friendly AI-powered PDF chatbot built using Retrieval-Augmented Generation (RAG).

## Features
- Upload PDF documents
- Extract text from PDFs
- Split documents into chunks
- Generate embeddings
- Retrieve relevant information using FAISS
- Ask questions and receive AI-generated answers

## Tech Stack
- Python
- Streamlit
- PyPDF
- Sentence Transformers
- FAISS
- Groq

## Run Locally

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Make sure Ollama is installed and the required model is available:

```bash
ollama pull llama3.2:1b
```

5. Run the application:

```bash
python -m streamlit run app.py
```

## Project Status
Learning project – actively developing and improving.