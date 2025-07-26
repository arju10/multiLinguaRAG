import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scripts.rag_api import rag_query

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def evaluate_groundedness(question, answer, contexts):
    answer_embedding = model.encode([answer])[0]
    context_embeddings = model.encode(contexts)
    similarities = cosine_similarity([answer_embedding], context_embeddings)[0]
    avg_similarity = np.mean(similarities) if similarities.size > 0 else 0.0
    return {
        "question": question,
        "answer": answer,
        "avg_groundedness_score": avg_similarity,
        "is_grounded": avg_similarity > 0.7
    }

def evaluate_relevance(question, contexts):
    question_embedding = model.encode([question])[0]
    context_embeddings = model.encode(contexts)
    similarities = cosine_similarity([question_embedding], context_embeddings)[0]
    avg_relevance = np.mean(similarities) if similarities.size > 0 else 0.0
    return {
        "question": question,
        "avg_relevance_score": avg_relevance,
        "is_relevant": avg_relevance > 0.7
    }

if __name__ == "__main__":
    test_cases = [
        {"question": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?", "expected_answer": "শুম্ভুনাথ"},
        {"question": "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?", "expected_answer": "মামাকে"},
        {"question": "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?", "expected_answer": "১৫ বছর"},
    ]

    for case in test_cases:
        result = rag_query(case["question"])
        groundedness = evaluate_groundedness(case["question"], result["answer"], result["contexts"])
        relevance = evaluate_relevance(case["question"], result["contexts"])
        print(f"Question: {case['question']}")
        print(f"Expected Answer: {case['expected_answer']}")
        print(f"Actual Answer: {result['answer']}")
        print(f"Groundedness: {groundedness['avg_groundedness_score']:.3f} ({groundedness['is_grounded']})")
        print(f"Relevance: {relevance['avg_relevance_score']:.3f} ({relevance['is_relevant']})")
        print("-" * 50)