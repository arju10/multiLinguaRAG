## ✅ **Phase‑3: Embedding & Indexing**

### 📌 **Objective**

In this phase, our goal was to:

* Load the **preprocessed chunks** from **MongoDB**.
* Convert each chunk into a **semantic vector** using the **OpenAI Embedding API**.
* Store all the vectors in **Weaviate**, an open‑source vector database, to enable efficient similarity search later.

---

### ⚙️ **Tools & Tech Used**

* **MongoDB:** Stores raw, cleaned, and chunked text data.
* **OpenAI Embedding API:** Generates high‑quality vector embeddings using the `text-embedding-3-large` model.
* **Weaviate:** An open‑source vector database that indexes and manages our vectors.

---

### 🚀 **How it works**

1️⃣ **Schema Creation**

* A Weaviate class named **`BanglaChunk`** was created.
* The class has a `content` property for the actual text.
* The `vectorizer` is set to **`none`**, since we are using external embeddings.

2️⃣ **Fetch Chunks & Generate Embeddings**

* The script reads all cleaned semantic chunks from MongoDB.
* Each chunk is passed to OpenAI’s Embedding API to create a vector representation.

3️⃣ **Upload to Weaviate**

* For each chunk, both the text and its vector are pushed to Weaviate.
* This makes it ready for similarity search in the next phase.

---

### ✅ **Why this step matters**

* It transforms raw text into a form that machines can compare **semantically**.
* We can later embed any user query and find the most similar chunks quickly.
* It sets up the foundation for our **Retriever** system.

---

### ⚡️ **Key Details**

* **Embedding Model:** `text-embedding-3-large`
* **Why this model?** It’s multilingual, powerful, and captures semantic meaning well.
* **Vector DB:** Weaviate is open‑source, easy to scale, and supports custom embeddings.

---

### 📁 **Output**

* All text chunks are now indexed in Weaviate.
* MongoDB connection is closed cleanly.
* The schema is automatically created if it doesn’t exist.

---

**📂 File:**
Example file: `scripts/weaviate_utils.py`

---
 The vector database is now ready to serve semantic search.

---

