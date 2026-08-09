# 🔨 InsightForge

> **Personal AI Knowledge Forge** — upload your documents, search them semantically, and ask questions using Retrieval-Augmented Generation (RAG) with Google Gemini.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-InsightForge-FF4B4B?logo=streamlit&logoColor=white)](https://insightforgefd.streamlit.app)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white)](https://github.com/ffrshdd/InsightForge)

---

## 🌐 Live Demo

🚀 **[Launch InsightForge](https://insightforgefd.streamlit.app)**

---

## 🧠 What is InsightForge?

InsightForge is an AI-powered document knowledge assistant built around a **Retrieval-Augmented Generation (RAG)** pipeline.

Instead of relying only on an LLM's general knowledge, InsightForge retrieves relevant information from the user's uploaded documents and provides that context to Google Gemini before generating an answer.

This makes it possible to ask questions about personal documents, notes, research papers, resumes, and other supported files.

---

## ✨ Features

- 📄 Upload **PDF, DOCX, TXT, Markdown, JSON, and CSV** files
- 🧠 Semantic search using **FAISS + Sentence Transformers**
- 🤖 AI-powered Q&A using **Google Gemini**
- 💬 Conversation-based chat interface
- 🔍 Search history
- 📊 Analytics dashboard with document and file-type statistics
- 📂 Document Manager with deletion and automatic re-indexing
- 📑 Source citations for retrieved document chunks
- ⬇️ Export answers as **TXT or Markdown**
- 🎨 Material-inspired dark user interface
- 🔐 API key managed through environment variables / Streamlit Secrets

---

## ⚙️ How It Works

```text
                User uploads document
                        │
                        ▼
                Document extraction
                        │
                        ▼
                  Text chunking
                        │
                        ▼
            Sentence Transformer embeddings
                        │
                        ▼
                   FAISS index
                        │
                        ▼
                 User asks a question
                        │
                        ▼
              Semantic similarity search
                        │
                        ▼
              Relevant document chunks
                        │
                        ▼
                  Google Gemini
                        │
                        ▼
              Grounded AI response
                        │
                        ▼
                  Source citations
