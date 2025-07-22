# MultiLinguaRAG

A Retrieval-Augmented Generation (RAG) system supporting multiple languages (Bangla, English, etc.).  
This project extracts text from PDFs, chunks them, creates embeddings, stores in a vector database, and serves answers via an LLM-powered API.

## Project Setup

### 0. Project Initialization
- Python virtual environment
- Required packages
- Git setup

### Upcoming Phases
- PDF extraction & cleaning
- Chunking & storage
- Embedding & indexing
- Query retrieval & generation
- API serving
- Deployment with Docker

### Folder Structure (Initial)
```bash
MultiLinguaRAG/
 ├── docs/               # Docs, notes, readme
 ├── data/               # Raw & cleaned data
 ├── scripts/            # Python scripts: extraction, chunking, indexing
 ├── app/                # FastAPI app, routes
 ├── embeddings/         # Embedding models, vector index files
 ├── config/             # .env, settings.py
 ├── docker/             # Dockerfile, Compose
 ├── requirements.txt
 ├── README.md
```