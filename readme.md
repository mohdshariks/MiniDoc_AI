
# 📚 MiniDoc AI – Intelligent PDF Assistant

Chat with your documents using AI and Retrieval-Augmented Generation (RAG).

MiniDoc AI is an AI-powered PDF question-answering application built with Python and Streamlit. Upload PDF documents, ask questions in natural language, and receive answers based on the document content.

## ✨ Features

- 📄 PDF Upload and Processing
- 🧠 Retrieval-Augmented Generation (RAG)
- 🔍 Semantic Search using Sentence Transformers and FAISS
- 📝 OCR support for scanned PDF documents
- ⚡ Batch PDF processing
- 💾 OCR result caching to avoid repeated processing
- 🤖 AI-powered answers using Groq API
- 💬 Multiple chat conversations
- 📚 Subject-based document organization
- 🌙 Modern Streamlit interface
- 📖 PDF document viewer

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application interface |
| PyMuPDF | PDF text extraction |
| Tesseract OCR | Scanned PDF text extraction |
| Pillow | Image processing |
| Sentence Transformers | Text embeddings |
| FAISS | Vector similarity search |
| BM25 | Keyword-based retrieval |
| NumPy | Numerical operations |
| Groq API | AI response generation |

## 🔄 How It Works

1. Upload PDF documents.
2. Extract text from digital PDFs.
3. Use OCR to extract text from scanned PDFs.
4. Split extracted text into smaller chunks.
5. Generate embeddings using Sentence Transformers.
6. Store embeddings in a FAISS vector index.
7. Retrieve relevant document chunks using semantic and keyword search.
8. Send retrieved context to the Groq API.
9. Generate answers based on the document content.

## 📂 Project Structure

```text
MiniDoc-AI/
│
├── app.py
├── config.py
├── pdf_processor.py
├── retrieval.py
├── llm.py
├── ui.py
│
├── requirements.txt
├── packages.txt
├── readme.md
├── .gitignore
│
├── .streamlit/
│   └── secrets.toml
│
└── .minidoc_cache/
```

Note: `.streamlit/secrets.toml`, `.minidoc_cache/`, and the virtual environment should not be uploaded to GitHub.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/mohdshariks/MiniDoc_AI.git
```

### 2. Navigate to the project directory

```bash
cd MiniDoc_AI
```

### 3. Create a virtual environment

```bash
python -m venv rag_env312
```

### 4. Activate the environment

Windows PowerShell:

```powershell
.\rag_env312\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Configure your Groq API key

Create a `.streamlit` folder and a `secrets.toml` file inside it.

Add:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

Get your API key from the [Groq Console](https://console.groq.com/).

### 7. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal.

## ☁️ Deployment

MiniDoc AI can be deployed using Streamlit Community Cloud.

1. Push the project to GitHub.
2. Visit [Streamlit Community Cloud](https://share.streamlit.io/).
3. Connect your GitHub repository.
4. Select `app.py` as the main file.
5. Open App Settings → Secrets.
6. Add your Groq API key:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

7. Save and deploy.

## 🔐 Security

- Never upload API keys to GitHub.
- Keep `.streamlit/secrets.toml` in `.gitignore`.
- Keep private documents and OCR cache files out of GitHub.
- Use Streamlit Cloud Secrets for deployed applications.

## 🚀 Future Improvements

- Advanced document summarization.
- Conversation export.
- Improved retrieval and reranking.
- Support for additional file formats.
- Enhanced document navigation.
- More advanced chat features.

## 👨‍💻 Author

**Mohammed Sharik**

GitHub: [@mohdshariks](https://github.com/mohdshariks)

Project: [MiniDoc AI](https://github.com/mohdshariks/MiniDoc_AI)

---

*MiniDoc AI – Making document understanding smarter with AI and RAG.*
