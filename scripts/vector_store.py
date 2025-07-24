# vector_store.py
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

load_dotenv()

# 1️⃣ Mongo
client = MongoClient(os.getenv("MONGO_URI"))
db = client["multilinguarag"]
collection = db["chunks"]

# 2️⃣ Load Chunks
doc = collection.find_one({"book_name": "HSC26 Bangla 1st Paper"})
chunks = doc["semantic_chunks"]
if not doc:
    raise ValueError("❌ No document found in MongoDB! Run preprocessing first.")

chunks = doc.get("semantic_chunks")
if not chunks:
    raise ValueError("❌ No semantic_chunks found in document!")
print(f"🔍 Loaded {len(chunks)} chunks.")

# 3️⃣ Embedding Model
print("⏳ Generating embeddings...")
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

embeddings = model.encode(chunks, show_progress_bar=True)
embeddings = np.array(embeddings).astype('float32')

# 4️⃣ FAISS Index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print(f"✅ FAISS index ready! Total vectors: {index.ntotal}")

# 5️⃣ Save Index + Chunks map
# faiss.write_index(index, "faiss_index.idx")

# with open("chunks.pkl", "wb") as f:
#     pickle.dump(chunks, f)

# print("✅ Saved index & chunks mapping!")

# client.close()

# 5️⃣ Save Index + Chunks map
os.makedirs("vectorstore", exist_ok=True)
faiss.write_index(index, "vectorstore/faiss_index.idx")

with open("vectorstore/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("✅ Saved index & chunks mapping in /vectorstore!")
client.close()