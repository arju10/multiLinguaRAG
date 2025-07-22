# 📚 Phase 2: Preprocessing & Chunking

## ✅ Goal

In this phase, I cleaned the raw text data and split it into smaller parts (chunks) for better semantic search and RAG performance.  
The steps were done in a single Python script (`preprocess_utils.py`).

---

## 🗂️ Steps Done

1. **Text Cleaning**
   - Removed unwanted symbols, numbers, noise lines.
   - Fixed common OCR mistakes in Bangla.
   - Normalized Bangla & English Unicode.
   - Removed single random characters and punctuation.

2. **Noise Removal**
   - Deleted page numbers, extra headers, question labels.

3. **Semantic Chunking**
   - Tried paragraph-based split first.
   - If no paragraphs found, used a fallback method: split text into ~300 character blocks.
   - Ensures all text is chunked properly, even if the PDF is messy.

4. **Save Data**
   - Saved cleaned text + chunks to MongoDB (`chunks` collection).
   - Saved local files:
     - `./data/cleaned_text_HSC26.txt`
     - `./data/chunks_HSC26.txt`

---

## 🧩 Chunking Strategy

**Method:**  
- Paragraph split (`\n\n`) → fallback to fixed size chunks.
- This helps keep related lines together.
- Works for both English & Bangla books.
- Makes embedding & vector DB storage easier.

**Why:**  
- Helps the RAG system find exact answers.
- Avoids big blocks that the model can't handle.
- Makes semantic search more accurate.

---

## 🗃️ Output Example

- ✅ Raw Length: `83416`
- ✅ Cleaned Length: `70276`
- ✅ Total Chunks: `235`

---

## ⚙️ Tools & Libraries

- Python
- Regex (`re`, `regex`)
- `pymongo` for MongoDB
- `dotenv` for environment variables

---

## 📌 Next Step

The cleaned & chunked text will be used in **Phase 3**, where we will:
- Generate embeddings
- Store them in a vector database
- Build the RAG retrieval system

---