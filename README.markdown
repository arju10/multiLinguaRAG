# Multilingual RAG System for HSC26 Bangla 1st Paper

## Overview
A Retrieval-Augmented Generation (RAG) system for answering queries in English and Bangla based on the "HSC26 Bangla 1st Paper" book. The system extracts text from a PDF (using OCR for Bangla, direct extraction for English), preprocesses it, chunks it, stores embeddings in a vector database, and generates answers using a lightweight LLM. It supports short-term memory (conversation history) and includes an evaluation module for groundedness and relevance.

## Setup Guide
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/arju10/multiLinguaRAG.git
   cd multiLinguaRAG
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Requirements include: `pymupdf`, `pymongo`, `python-dotenv`, `sentence-transformers`, `faiss-cpu`, `langchain`, `transformers`, `torch`, `fastapi`, `uvicorn`, `scikit-learn`, `numpy`, `pytesseract`, `Pillow` etc.
3. **Install Tesseract OCR**:
   - Ubuntu: `sudo apt-get install tesseract-ocr tesseract-ocr-ben`
   - Windows: Install from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki) and add to PATH.
   - macOS: `brew install tesseract tesseract-lang`
4. **Set Up Environment Variables**:
   Create a `.env` file in the root directory:
   ```bash
   MONGO_URI=<your-mongodb-uri>
   ```
5. **Prepare the PDF**:
   Place the `HSC26-Bangla1st-Paper.pdf` file in the `./data/` folder.
6. **Run the Pipeline**:
   ```bash
   python scripts/extract_and_clean.py
   python scripts/preprocess_utils.py
   python scripts/vector_store.py
   python scripts/rag_api.py
   ```
7. **Start the API**:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```
8. **Evaluate the System**:
   ```bash
   python scripts/evaluate_rag.py
   ```

## Tools, Libraries, and Packages
- **Text Extraction**: `PyMuPDF` (direct PDF extraction), `pytesseract`, `Pillow` (OCR for Bangla)
- **Preprocessing**: `unicodedata`, `re`, `langchain`
- **Embedding**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Vector Store**: `faiss-cpu`
- **Database**: `pymongo` (MongoDB Atlas)
- **LLM**: `Qwen/Qwen2-0.5B-Instruct`
- **API**: `fastapi`, `uvicorn`
- **Evaluation**: `scikit-learn`, `numpy`

## Sample Queries and Outputs
- **Query**: অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?
  - **Output**: শুম্ভুনাথ
- **Query**: কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?
  - **Output**: মামাকে
- **Query**: বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?
  - **Output**: ১৫ বছর

## API Documentation
- **Endpoint**: `/query`
  - **Method**: POST
  - **Request Body**: `{ "question": "<user question in Bangla or English>" }`
  - **Response**: `{ "answer": "<generated answer>", "contexts": ["<chunk1>", "<chunk2>", ...], "language": "bn|en" }`
  - **Example**:
    ```bash
    curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"question": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"}'
    ```
  - **Response Example**:
    ```json
    {
      "answer": "শুম্ভুনাথ",
      "contexts": ["<relevant chunk>", ...],
      "language": "bn"
    }
    ```
- **Endpoint**: `/health`
  - **Method**: GET
  - **Response**: `{ "status": "ok" }`

## Evaluation Matrix
| Question | Answer | Groundedness Score | Is Grounded? | Relevance Score | Is Relevant? |
|----------|--------|--------------------|--------------|----------------|-------------|
| অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে? | শুম্ভুনাথ | 0.85 | Yes | 0.82 | Yes |
| কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে? | মামাকে | 0.80 | Yes | 0.79 | Yes |
| বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল? | ১৫ বছর | 0.78 | Yes | 0.76 | Yes |

## Answers to Required Questions
1. **What method or library did you use to extract the text, and why? Did you face any formatting challenges with the PDF content?**
 - **Method/Library:** Used PyMuPDF for direct text extraction (English) and pytesseract with Pillow for OCR-based extraction (Bangla). Direct extraction is used for English due to its accuracy with standard fonts. OCR is used for Bangla to handle cases where text is embedded as images or uses non-standard fonts. Image preprocessing (grayscale, contrast enhancement, sharpening) is applied to improve OCR accuracy.

- **Formatting Challenges:** Bangla OCR may misinterpret characters due to low image quality or complex fonts, leading to noisy text. This is mitigated by preprocessing steps (noise removal, whitespace cleaning) and image enhancement. English extraction is cleaner but may struggle with tables or footnotes, handled by regex-based noise removal.

2. **What chunking strategy did you choose? Why do you think it works well for semantic retrieval?**
   - **Strategy**: Hybrid chunking (paragraph → sentence) using LangChain’s `RecursiveCharacterTextSplitter` with a chunk size of 300 and overlap of 50.
   - **Why**: Preserves semantic coherence by prioritizing paragraph boundaries while ensuring chunks are small enough for precise retrieval. The overlap ensures context continuity.
   
3. **What embedding model did you use? Why did you choose it? How does it capture the meaning of the text?**
   - **Model**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
   - **Why**: Chosen for its multilingual support (English and Bangla) and lightweight architecture, suitable for semantic similarity tasks.
   - **How**: Uses a transformer-based model to encode text into 384-dimensional vectors, capturing semantic relationships through training on paraphrase and similarity datasets.
4. **How are you comparing the query with your stored chunks? Why did you choose this similarity method and storage setup?**
   - **Comparison**: Cosine similarity via FAISS `IndexFlatIP`.
   - **Why**: Cosine similarity is robust for semantic comparison as it normalizes vector magnitudes. FAISS is efficient for vector search, and MongoDB stores metadata and raw text for flexibility.
5. **How do you ensure that the question and the document chunks are compared meaningfully? What would happen if the query is vague or missing context?**
   - **Ensuring Meaningful Comparison**: The multilingual embedding model encodes queries and chunks into a shared semantic space. A similarity threshold (0.5) filters out irrelevant chunks.
   - **Vague Queries**: May retrieve less relevant chunks, leading to "Answer not found" or incorrect answers. A re-ranking step or query clarification could improve this.
6. **Do the results seem relevant? If not, what might improve them?**
   - **Relevance**: Results are relevant for the test cases (see evaluation matrix). Improvements include using a larger embedding model (`mpnet-base-v2`), re-ranking with a cross-encoder, or expanding the document corpus.