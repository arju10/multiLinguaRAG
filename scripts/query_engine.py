# query_engine.py
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Load saved index
# index = faiss.read_index("vectorstore/faiss_index.idx")
# with open("vectorstore/chunks.pkl", "rb") as f:
#     chunks = pickle.load(f)
try:
    index = faiss.read_index("vectorstore/faiss_index.idx")
    with open("vectorstore/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
except Exception as e:
    raise FileNotFoundError(f"❌ Could not load index or chunks. Run vector_store.py first.\n{e}")


model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


def search(query: str, top_k: int = 3):
    q_embed = model.encode([query]).astype('float32')
    distances, indices = index.search(q_embed, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1:  # FAISS returns -1 if no result
            continue
        results.append({
            "chunk": chunks[idx],
            "similarity": float(1 / (1 + distances[0][i]))  # Convert L2 to approximate similarity
        })
    return results
# def search(query: str, top_k: int = 3):
#     q_embed = model.encode([query]).astype('float32')
#     distances, indices = index.search(q_embed, top_k)

#     results = []
#     for i, idx in enumerate(indices[0]):
#         results.append({
#             "chunk": chunks[idx],
#             "distance": float(distances[0][i])
#         })
#     return results

if __name__ == "__main__":
    # q = input("🔍 Enter your question: ")
    # hits = search(q)
    # for h in hits:
    #     print("\n", h)
        print("🔍 RAG Query Engine Ready (Type 'quit' to exit)\n")
        while True:
            q = input("❓ Question: ").strip()
            if q.lower() in {"quit", "exit", "বাই", "বন্ধ করো"}:
                break
            if not q:
                continue

            hits = search(q)
            print("\n📄 Retrieved Chunks:")
            for i, h in enumerate(hits):
                print(f"\n[{i+1}] (Similarity: {h['similarity']:.3f})\n{h['chunk']}\n{'-'*50}")
