# 📚 MiniDoc AI – Intelligent PDF Assistant

**Chat with your documents using AI and Retrieval-Augmented Generation (RAG).**

MiniDoc AI is an AI-powered PDF question-answering application built with Python and Streamlit. Upload a PDF, ask questions in natural language, and receive answers grounded in the document's content with relevant source references.

## ✨ Features

- 📄 **PDF Upload:** Upload and process PDF documents.
- 🧠 **RAG Pipeline:** Retrieve relevant document content before generating answers.
- 🔍 **Semantic Search:** Find relevant text using Sentence Transformers and FAISS.
- 🤖 **AI-Powered Answers:** Generate responses using the Groq API.
- 📚 **Source References:** View retrieved text and page numbers.
- 💬 **Chat History:** Create and switch between conversations.
- 🌙 **Dark SaaS Interface:** Modern, responsive Streamlit UI.
- 📱 **Responsive Design:** Designed to work on desktop and mobile.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application interface |
| PyPDF | PDF text extraction |
| Sentence Transformers | Text embeddings |
| FAISS | Vector similarity search |
| Groq API | AI response generation |
| NumPy | Numerical operations |

## 🔄 How It Works

1. Upload a PDF document.
2. Extract text from the PDF.
3. Split the extracted text into smaller chunks.
4. Generate embeddings using Sentence Transformers.
5. Store embeddings in a FAISS vector index.
6. Convert the user's question into an embedding.
7. Retrieve the most relevant document chunks.
8. Send the retrieved context to the Groq API.
9. Display the generated answer and source references.

## 📂 Project Structure

```text
MiniDoc-AI/
│
├── app.py
├── requirements.txt
├── readme.md
├── .gitignore
│
└── .streamlit/
    └── secrets.toml
```

**Note:** `secrets.toml` is used for local development and should never be uploaded to GitHub.

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

**Windows PowerShell:**

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

Get an API key from the [Groq Console](https://console.groq.com/).

### 7. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal.

## ☁️ Deployment

MiniDoc AI can be deployed using Streamlit Community Cloud.

1. Push the project to GitHub.
2. Visit [Streamlit Community Cloud](https://share.streamlit.io/).
3. Connect your GitHub repository.
4. Select `app.py` as the main file.
5. Add your API key under **App Settings → Secrets**.

Add this secret:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

6. Save and deploy.

## 🔐 Security

- Never upload API keys to GitHub.
- Keep `.streamlit/secrets.toml` in `.gitignore`.
- Use Streamlit Cloud Secrets for deployed applications.
- Do not share private API credentials.

## 🚀 Future Improvements

- Support for multiple document uploads.
- Document summarization.
- Conversation export.
- Improved retrieval and reranking.
- Support for additional file formats.
- More advanced chat features.

## 👨‍💻 Author

**Mohammed Sharik**

GitHub: [@mohdshariks](https://github.com/mohdshariks)

Project: [MiniDoc AI](https://github.com/mohdshariks/MiniDoc_AI)

---

*MiniDoc AI – Making document understanding smarter with AI and RAG.*
