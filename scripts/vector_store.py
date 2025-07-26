import os
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

load_dotenv()
client = MongoClient(os.getenv("MONGO_URI"))
db = client["multilinguarag"]
collection = db["chunks"]

# Load Chunks
doc = collection.find_one({"book_name": "HSC26 Bangla 1st Paper"})
if not doc:
    raise ValueError("❌ No document found in MongoDB! Run preprocessing first.")

chunks = doc.get("semantic_chunks")
if not chunks:
    raise ValueError("❌ No semantic_chunks found in document!")
print(f"🔍 Loaded {len(chunks)} chunks.")

chunk_lengths = [len(c) for c in chunks]
print(f"Max: {max(chunk_lengths)} | Min: {min(chunk_lengths)} | Avg: {sum(chunk_lengths)/len(chunk_lengths):.2f}")

# Embedding Model
print("⏳ Generating embeddings...")
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
embeddings = model.encode(chunks, show_progress_bar=True)
embeddings = np.array(embeddings).astype('float32')

# FAISS Index (Cosine Similarity)
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
faiss.normalize_L2(embeddings)
index.add(embeddings)

print(f"✅ FAISS index ready! Total vectors: {index.ntotal}")

# Save Index + Chunks Map
os.makedirs("vectorstore", exist_ok=True)
faiss.write_index(index, "vectorstore/faiss_index.idx")
with open("vectorstore/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("✅ Saved index & chunks mapping in /vectorstore!")
client.close()