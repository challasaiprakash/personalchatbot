# 🧠 AskSai — AI Resume Chatbot

> An AI-powered chatbot that answers questions about my professional profile using RAG (Retrieval-Augmented Generation) and LangChain.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![LangChain](https://img.shields.io/badge/LangChain-RAG-brightgreen?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-red?style=flat-square)
![Groq](https://img.shields.io/badge/LLM-Groq%20gpt--oss--20b-orange?style=flat-square)
![FastEmbed](https://img.shields.io/badge/Embeddings-FastEmbed-teal?style=flat-square)
![ChromaDB](https://img.shields.io/badge/VectorStore-ChromaDB-purple?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## 🚀 Live Demo

👉 **[Try AskSai here](https://personalchatbot-hjnhtqk8ej6a6jx7vbnaho.streamlit.app/)**

---

## 💡 What Is AskSai?

AskSai is a conversational AI chatbot that acts as a smart, interactive version of my resume. Instead of scrolling through a PDF, anyone — recruiters, collaborators, or curious visitors — can ask questions in plain English and get instant answers about my:

- 🔬 **Research & Publications**
- 🛠️ **Projects & Tech Stack** (10 projects across computer vision, audio ML, NLP, forecasting, and LLM apps)
- 💼 **Work Experience**
- 🎓 **Education**
- 🧰 **Skills & Tools**
- 🎯 **Career Interests**
- 📬 **Contact Information**

Built to showcase a real-world, end-to-end RAG pipeline using open-source tools.

### 💬 Try asking

- *"What is Sai's PhD research about?"*
- *"Has Sai worked on computer vision projects with industry?"*
- *"Tell me about his Scientific Reports paper."*
- *"What results did the thermal storage forecasting project achieve?"*
- *"What roles is Sai open to?"*

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
       (FastEmbed embeddings)
                  │
                  ▼
         Intent Router
         (Project / Experience / General)
                  │
                  ▼
       openai/gpt-oss-20b
       (via Groq API)
                  │
                  ▼
          AskSai Response
```

---

## ⚙️ Tech Stack

| Layer            | Technology                                      |
|------------------|-------------------------------------------------|
| Framework        | LangChain (Core, Text Splitters, Runnables)     |
| LLM              | openai/gpt-oss-20b (Groq API)                   |
| Embeddings       | FastEmbed (lightweight ONNX embeddings)         |
| Vector Store     | ChromaDB                                        |
| UI               | Streamlit                                       |
| Out-of-scope Log | Google Sheets (gspread + GCP Service Account)   |
| Deployment       | Streamlit Cloud                                 |

---

## ✨ Key Features

### 🔍 RAG Pipeline
Retrieves the most semantically relevant chunks from the resume before passing context to the LLM, so answers stay grounded in what the resume actually says.

### 🧭 Intent-Aware Query Routing
A custom query classifier detects whether the user is asking about **projects**, **work experience**, or **general profile info** and routes to the matching document subset for sharper, more relevant answers.

### 🔤 Fallback Lexical Retriever
If semantic retrieval is unavailable, the system falls back to a keyword-based BM25-style retriever, so the app keeps answering.

### 📋 Out-of-Scope Detection + Auto-Logging
When a question falls outside the resume, AskSai politely redirects the user and **automatically logs the question to a Google Sheet**, so I can review gaps and keep improving coverage.

### 💬 Streamlit Chat UI
Clean, interactive chat interface with persistent session history, custom CSS styling, and an animated alien mascot 👽 that walks across the input bar while idle.

---

## 🔧 Local Setup

### 1. Clone the repo

```bash
git clone https://github.com/challasaiprakash/personalchatbot.git
cd personalchatbot
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

> **Updating the bot's knowledge:** replace `resume_and_projects.txt` and restart the app (or clear the Streamlit cache) so the vector store is rebuilt from the new file.

### 4. Configure environment variables

Create a `.env` file in the project root:

Get a free API key at [console.groq.com](https://console.groq.com/).

```env
GROQ_API_KEY=your_groq_api_key_here
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
GROQ_API_KEY = "your_groq_api_key"
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
personalchatbot/
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
langchain-groq
langchain-community
langchain-chroma
chromadb
fastembed
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

This project demonstrates a **practical, end-to-end RAG implementation** using open-source tools, covering the full pipeline from document ingestion and semantic chunking to vector retrieval, intent routing, and LLM response generation, applied to a real personal use case.

It's also a living portfolio piece: as my experience grows, I update the resume file and the chatbot's knowledge grows with it.

---

## 🙋 About Me

Built by **Ing. Sai Prakash Challa**, PhD researcher in machine learning and computer vision at VSB – Technical University of Ostrava, Czechia.

- 🔗 [LinkedIn](https://www.linkedin.com/in/challasp/)
- 🌐 [Portfolio](https://tinyurl.com/portfoliochalla)
- 📚 [My other app: Multi-Document Research Assistant](https://researchassistantv2-79aaycvac74meqppdjn92e.streamlit.app/)
- 📧 [challasaiprakashcsp@gmail.com](mailto:challasaiprakashcsp@gmail.com)

---

*If you found this useful, drop a ⭐ on the repo!*
