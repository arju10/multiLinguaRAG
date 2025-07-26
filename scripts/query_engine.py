import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    index = faiss.read_index("vectorstore/faiss_index.idx")
    with open("vectorstore/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
except Exception as e:
    raise FileNotFoundError(f"❌ Could not load index or chunks. Run vector_store.py first.\n{e}")

embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def search(query: str, top_k: int = 3, similarity_threshold: float = 0.5):
    query_embedding = embedding_model.encode([query]).astype('float32')
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        similarity = float(distances[0][i])
        if similarity < similarity_threshold:
            continue
        results.append({
            "chunk": chunks[idx],
            "similarity": similarity
        })
    return results

if __name__ == "__main__":
    print("🔍 RAG Query Engine Ready (Type 'quit' to exit)\n")
    while True:
        q = input("❓ Question: ").strip()
        if q.lower() in {"quit", "exit", "বাই", "বন্ধ করো"}:
            print("Thank you.")
            break
        if not q:
            continue
        hits = search(q)
        if not hits:
            print("❌ No similar chunks found.")
            continue
        print("\n📄 Retrieved Chunks:")
        for i, h in enumerate(hits):
            print(f"\n[{i+1}] (Similarity: {h['similarity']:.3f})\n{h['chunk']}\n{'-'*50}")