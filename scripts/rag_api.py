# rag_api.py
# import os
# from dotenv import load_dotenv
# from fastapi import FastAPI
# from pydantic import BaseModel
# from query_engine import search
# import requests
# load_dotenv()

# app = FastAPI()

# class Query(BaseModel):
#     question: str

# @app.post("/query")
# def answer(query: Query):
#     hits = search(query.question)
#     context = " ".join([h["chunk"] for h in hits])

#     ollama_url = "http://localhost:11434/api/generate"
#     payload = {
#         "model": "llama3:8b",
#         "prompt": f"Question: {query.question}\nContext: {context}\nAnswer in Bangla:"
#     }

#     r = requests.post(ollama_url, json=payload)
#     print(r.status_code)
#     print(r.text)

#     if r.status_code != 200:
#         return {"error": "Ollama failed", "details": r.text}

#     output = r.json()
#     answer = output.get("response", "No valid answer")

#     return {
#         "question": query.question,
#         "matches": hits,
#         "final_answer": answer
#     }


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("rag_api:app", host="0.0.0.0", port=8000, reload=True)
    



# ONe test
# # rag_pipeline.py
# from query_engine import search
# import ollama  # pip install ollama

# def rag_query(question: str, top_k: int = 3):
#     # Retrieve
#     results = search(question, top_k=top_k)
#     context = "\n\n".join([r["chunk"] for r in results])

#     # Detect language
#     lang = "bn" if any('\u0980' <= c <= '\u09FF' for c in question) else "en"

#     # Prompt
#     if lang == "bn":
#         prompt = f"""
#         প্রসঙ্গ: {context}
#         প্রশ্ন: {question}
#         উত্তর (সংক্ষেপে):
#         """
#     else:
#         prompt = f"""
#         Context: {context}
#         Question: {question}
#         Answer (briefly):
#         """

#     # Generate
#     response = ollama.generate(model="llama3:8b", prompt=prompt)
#     return {
#         "answer": response["response"].strip(),
#         "contexts": [r["chunk"] for r in results]
#     }

# # Test
# if __name__ == "__main__":
#     q = "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
#     result = rag_query(q)
#     print("✅ Answer:", result["answer"])



# test-2
# pip install transformers torch accelerate bitsandbytes sentencepiece
# rag_pipeline.py
from query_engine import search  # 
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os


# --- Load Lightweight LLM (4-bit quantized) ---
# model_id = "google/gemma-2b-it"

# model_id="tiiuae/falcon-rw-1b"
model_id="microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,  # Save memory
    device_map="auto",           # Use CPU if no GPU
    load_in_4bit=True,           # 4-bit quantization (critical for low RAM)
    low_cpu_mem_usage=True
)

# Create text generation pipeline
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=150,
    temperature=0.4,
    top_p=0.9,
    do_sample=True,
)

def generate_answer(context: str, question: str, lang: str = "bn"):
    if lang == "bn":
        prompt = f"""
        প্রসঙ্গ: {context}
        প্রশ্ন: {question}
        উত্তর (সংক্ষেপে):
        """
    else:
        prompt = f"""
        Context: {context}
        Question: {question}
        Answer (briefly):
        """

    # Generate answer
    sequences = llm_pipeline(prompt)
    return sequences[0]['generated_text'][len(prompt):].strip()

def rag_query(question: str, top_k: int = 3):
    # Detect language
    lang = "bn" if any('\u0980' <= c <= '\u09FF' for c in question) else "en"

    # Retrieve from FAISS
    results = search(question, top_k=top_k)
    context = "\n\n".join([r["chunk"] for r in results])

    # Generate answer
    try:
        answer = generate_answer(context, question, lang)
    except Exception as e:
        answer = "❌ উত্তর তৈরি করতে সমস্যা হয়েছে। মডেল লোড করুন।"

    return {
        "answer": answer,
        "contexts": [r["chunk"] for r in results],
        "language": lang
    }

# Test
if __name__ == "__main__":
    q = "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
    result = rag_query(q)
    print("✅ Answer:", result["answer"])