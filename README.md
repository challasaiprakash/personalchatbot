# 🧠 AskSai — AI Resume Chatbot


> An AI-powered chatbot that answers questions about my professional profile using RAG (Retrieval-Augmented Generation) and LangChain.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![LangChain](https://img.shields.io/badge/LangChain-RAG-brightgreen?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-red?style=flat-square)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Qwen2.5--7B-yellow?style=flat-square)
![ChromaDB](https://img.shields.io/badge/VectorStore-ChromaDB-purple?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## 🚀 Live Demo

👉 **[Try AskSai here](your-streamlit-app-link)**

---

## 💡 What Is AskSai?

AskSai is a conversational AI chatbot that acts as a smart, interactive version of my resume. Instead of scrolling through a PDF, anyone — recruiters, collaborators, or curious visitors — can simply ask questions in plain English and get instant, accurate answers about my:

- 🛠️ **Projects & Tech Stack**
- 💼 **Work Experience**
- 🎓 **Education**
- 📬 **Contact Information**
- 🧰 **Skills & Tools**

Built to showcase a real-world, end-to-end RAG pipeline using open-source tools.

---

## 🏗️ Architecture

```
Resume (.txt file)
       │
       ▼
 Document Ingestion
       │
       ├──► Project Docs        (custom section extractor)
       ├──► Experience Docs     (section-aware line parser)
       └──► General Splits      (RecursiveCharacterTextSplitter, chunk=1500)
                  │
                  ▼
       ChromaDB Vector Store
       (sentence-transformers/all-MiniLM-L6-v2 embeddings)
                  │
                  ▼
         Intent Router
         (Project / Experience / General)
                  │
                  ▼
       Qwen2.5-7B-Instruct
       (via HuggingFace Endpoint)
                  │
                  ▼
          AskSai Response
```

---

## ⚙️ Tech Stack

| Layer            | Technology                                      |
|------------------|-------------------------------------------------|
| Framework        | LangChain (Core, Text Splitters, Runnables)     |
| LLM              | Qwen2.5-7B-Instruct (HuggingFace Endpoint)      |
| Embeddings       | sentence-transformers/all-MiniLM-L6-v2          |
| Vector Store     | ChromaDB                                        |
| UI               | Streamlit                                       |
| Out-of-scope Log | Google Sheets (gspread + GCP Service Account)   |
| Deployment       | Streamlit Cloud                                 |

---

## ✨ Key Features

### 🔍 RAG Pipeline
Retrieves the most semantically relevant chunks from the resume before passing context to the LLM — keeping responses accurate, grounded, and hallucination-free.

### 🧭 Intent-Aware Query Routing
A custom query classifier detects whether the user is asking about **projects**, **work experience**, or **general profile info** and routes to the appropriate document subset for sharper, more relevant answers.

### 🔤 Fallback Lexical Retriever
If semantic embeddings are unavailable (e.g., API quota exceeded), the system falls back to a keyword-based BM25-style retriever — ensuring the app never goes down.

### 📋 Out-of-Scope Detection + Auto-Logging
When a question falls outside the resume context, AskSai gracefully redirects the user and **automatically logs the question to a Google Sheet** — so I can review gaps and continuously improve coverage.

### 💬 Streamlit Chat UI
Clean, interactive chat interface with persistent session history, custom CSS styling, and an animated alien mascot 👽 that walks across the input bar while idle.

---

## 🔧 Local Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/asksai.git
cd asksai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your resume

Place your resume as a `.txt` file inside the `data/` folder:

```
data/
└── resume_and_projects.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
GOOGLE_SHEET_ID=your_google_sheet_id_here        # optional, for out-of-scope logging
```

For Google Sheets logging, also place your GCP service account JSON at:

```
service_account.json
```

### 5. Run the app

```bash
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Cloud

1. Push the repo to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) and connect your repo
3. Add secrets in the Streamlit Cloud dashboard under **Settings → Secrets**:

```toml
HUGGINGFACEHUB_API_TOKEN = "your_token"
GOOGLE_SHEET_ID = "your_sheet_id"

[gcp_service_account]
type = "service_account"
project_id = "..."
private_key_id = "..."
private_key = "..."
client_email = "..."
# ... rest of your service account fields
```

---

## 📁 Project Structure

```
asksai/
│
├── app.py                        # Main Streamlit application
├── data/
│   └── resume_and_projects.txt   # Your resume (plain text)
├── service_account.json          # GCP credentials — local only, never commit!
├── .env                          # Secrets — local only, never commit!
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 📦 Requirements

```
streamlit
langchain
langchain-core
langchain-text-splitters
langchain-huggingface
langchain-chroma
chromadb
sentence-transformers
huggingface-hub
gspread
google-auth
python-dotenv
```

---

## 🛡️ .gitignore Recommendation

Make sure these are in your `.gitignore` to avoid leaking secrets:

```
.env
service_account.json
__pycache__/
*.pyc
.chromadb/
```

---

## 📌 Why I Built This

This project was built to demonstrate a **practical, end-to-end RAG implementation** using open-source tools — covering the full pipeline from document ingestion and semantic chunking, to vector retrieval, intent routing, and LLM response generation — applied to a real-world personal use case.

It's also a living portfolio piece: as my experience grows, the chatbot's knowledge updates automatically with the resume file.

---

## 🙋 About Me

Built by **Sai**

- 🔗 [LinkedIn](www.linkedin.com/in/challasp)
- 📧 your-CHALLASAIPRAKASHCSP@MAIL.COM

---

*If you found this useful, drop a ⭐ on the repo!*


