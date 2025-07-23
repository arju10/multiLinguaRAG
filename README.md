# MultiLinguaRAG

A Retrieval-Augmented Generation (RAG) system supporting multiple languages (Bangla, English, etc.).  
This project extracts text from PDFs, chunks them, creates embeddings, stores in a vector database, and serves answers via an LLM-powered API.

## Project Setup

### 0. Project Initialization
- Python virtual environment
- Required packages
- Git setup

## 📌 Questions:
### 1. What method or library did you use to extract the text, and why? Did you face any formatting challenges with the PDF content?**

**Answer:** 
I used pytesseract (Tesseract OCR) along with Pillow to extract the text. Although the original PDF looked selectable, its formatting was broken — so direct text extraction did not work properly for Bangla text.

To fix this, I first converted each page of the PDF into separate images. Then, I ran Tesseract OCR on those images to accurately extract the Bangla text.

This extra step helped to handle formatting issues in the PDF and ensured I got usable, editable text for the next processing phases.

### What chunking strategy did you choose (e.g. paragraph-based, sentence-based, character limit)? Why do you think it works well for semantic retrieval?

**Answer:** 
**Chunking Strategy:**
I used a simple and safe chunking method. First, I tried to split the text by paragraphs if there were clear newlines (\n\n). But sometimes Bangla PDFs or books don’t have clear paragraph breaks. So if paragraphs are not found or very few, I split the full text into smaller parts by a fixed length (about 300 characters).

**Why it works well:**
This way, no part is too big for the AI model to understand. Also, related sentences stay together, so meaning does not break. If the text has no proper newlines or punctuation, still we get enough chunks for good semantic search and answer generation.

This simple fallback keeps my RAG system accurate and safe for both Bangla and English data.

### Upcoming Phases
- PDF extraction & cleaning (Done)
- Chunking & storage (Done)
- Embedding & indexing
- Query retrieval & generation
- API serving
- Deployment with Docker

### Folder Structure (Initial)
```bash
MultiLinguaRAG/
 ├── docs/               # Docs, notes, readme
 ├── data/
        ├── images/              
 ├── scripts/            # Python scripts: extraction, chunking, indexing
        ├── extract_and_clean.py  # Raw & cleaned data          
 ├── app/                # FastAPI app, routes
 ├── embeddings/         # Embedding models, vector index files
 ├── config/             # .env, settings.py
 ├── docker/             # Dockerfile, Compose
 ├── requirements.txt
 ├── README.md
```
## 📌 Fully Open-Source Flow
``` 
PDF (PyMuPDF)
    ↓
Chunking (LangChain)
    ↓
Embeddings (Sentence Transformers, local)
    ↓
Vector DB (FAISS/Qdrant)
    ↓
Query encode (Sentence Transformers)
    ↓
Similarity Search (FAISS/Qdrant)
    ↓
LLM Answer (Ollama local)
    ↓
FastAPI Serve
```