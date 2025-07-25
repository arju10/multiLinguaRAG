# from query_engine import search  # 
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# import torch
# import os
# from dotenv import load_dotenv

# # --- Load Lightweight LLM (4-bit quantized) ---
# # model_id = "google/gemma-2b-it"

# model_id="tiiuae/falcon-rw-1b"

# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     torch_dtype=torch.float16,  # Save memory
#     device_map="auto",           # Use CPU if no GPU
#     load_in_4bit=True,           # 4-bit quantization (critical for low RAM)
#     low_cpu_mem_usage=True
# )

# # Create text generation pipeline
# llm_pipeline = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=150,
#     temperature=0.4,
#     top_p=0.9,
#     do_sample=True,
# )

# def generate_answer(context: str, question: str, lang: str = "bn"):
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

#     # Generate answer
#     sequences = llm_pipeline(prompt)
#     return sequences[0]['generated_text'][len(prompt):].strip()

# def rag_query(question: str, top_k: int = 3):
#     # Detect language
#     lang = "bn" if any('\u0980' <= c <= '\u09FF' for c in question) else "en"

#     # Retrieve from FAISS
#     results = search(question, top_k=top_k)
#     context = "\n\n".join([r["chunk"] for r in results])

#     # Generate answer
#     try:
#         answer = generate_answer(context, question, lang)
#     except Exception as e:
#         answer = "❌ উত্তর তৈরি করতে সমস্যা হয়েছে। মডেল লোড করুন।"

#     return {
#         "answer": answer,
#         "contexts": [r["chunk"] for r in results],
#         "language": lang
#     }

# # Test
# if __name__ == "__main__":
#     q = "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
#     result = rag_query(q)
#     print("✅ Answer:", result["answer"])



# import os
# from dotenv import load_dotenv
# from query_engine import search  # Your vector search stays local
# import requests

# load_dotenv()

# HF_TOKEN = os.getenv("HF_TOKEN")
# # print("HF_TOKEN",HF_TOKEN)
# MODEL_ID ="Qwen/Qwen2-0.5B-Instruct"

# API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"


# HEADERS = {
#     "Authorization": f"Bearer {HF_TOKEN}"
# }
# print("Headers:",HEADERS)


# def generate_answer(context: str, question: str, lang: str = "bn"):
#     if lang == "bn":
#         prompt = f"""
#         নিম্নলিখিত প্রসঙ্গ থেকে প্রশ্নের উত্তর দিন। তথ্য না থাকলে বলুন "আমি জানি না।"

#         প্রসঙ্গ: {context}
#         প্রশ্ন: {question}
#         উত্তর (সংক্ষেপে):
#         """
#     else:
#         prompt = f"""
#         Answer the question based on the context below. If not found, say "I don't know."

#         Context: {context}
#         Question: {question}
#         Answer (briefly):
#         """

#     payload = {
#         "inputs": prompt,
#         "parameters": {
#             "max_new_tokens": 150,
#             "temperature": 0.4,
#             "top_p": 0.9,
#             "do_sample": True
#         }
#     }

#     response = requests.post(API_URL, headers=HEADERS, json=payload)

#     if response.status_code == 200:
#         generated = response.json()
#         text = generated[0]["generated_text"]
#         return text[len(prompt):].strip()
#     else:
#         print("⚠️ HF API Error:", response.status_code, response.text)
#         return "❌ HF API problem"


# def rag_query(question: str, top_k: int = 3):
#     lang = "bn" if any('\u0980' <= c <= '\u09FF' for c in question) else "en"

#     results = search(question, top_k=top_k)
#     context = "\n\n".join([r["chunk"] for r in results])

#     try:
#         answer = generate_answer(context, question, lang)
#     except Exception as e:
#         print("Exception:", e)
#         answer = "❌ উত্তর তৈরি করতে সমস্যা হয়েছে।"

#     return {
#         "answer": answer,
#         "contexts": [r["chunk"] for r in results],
#         "language": lang
#     }


# if __name__ == "__main__":
#     q = "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
#     result = rag_query(q)
#     print("✅ Answer:", result["answer"])





# # rag_query.py
# import os
# from dotenv import load_dotenv
# from query_engine import search  # Your FAISS search
# import requests

# load_dotenv()

# HF_TOKEN = os.getenv("HF_TOKEN")  # Make sure this matches your .env
# # MODEL_ID = "google/gemma-2b-it"  # ✅ Text generation model
# # MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"
# # MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
# # MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"

# # ✅ Correct URL (no extra spaces!)
# # API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
# API_URL="https://router.huggingface.co/v1/chat/completions/HuggingFaceH4/zephyr-7b-beta:featherless-ai"

# HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# def generate_answer(context: str, question: str, lang: str = "bn"):
#     # Build prompt
#     if lang == "bn":
#         prompt = f"""
#         নিম্নলিখিত প্রসঙ্গ থেকে প্রশ্নের উত্তর দিন। তথ্য না থাকলে বলুন "আমি জানি না।"

#         প্রসঙ্গ:
#         {context}

#         প্রশ্ন: {question}
#         উত্তর:
#         """
#     else:
#         prompt = f"""
#         Answer the question based on the context below. If not found, say "I don't know."

#         Context:
#         {context}

#         Question: {question}
#         Answer:
#         """

#     # Payload
#     payload = {
#         "inputs": prompt,
#         "parameters": {
#             "max_new_tokens": 150,
#             "temperature": 0.4,
#             "top_p": 0.9,
#             "do_sample": True
#         }
#     }

#     try:
#         response = requests.post(API_URL, headers=HEADERS, json=payload)
#         response.raise_for_status()  # Will raise error for 4xx/5xx

#         output = response.json()
#         generated_text = output[0]["generated_text"]

#         # Extract only the answer (remove prompt echo)
#         answer = generated_text[len(prompt):].strip()
#         return answer if answer else "❌ উত্তর পাওয়া যায়নি।"

#     except requests.exceptions.HTTPError as http_err:
#         return f"❌ HTTP Error: {response.status_code} - {response.text}"
#     except Exception as e:
#         return f"❌ Error: {str(e)}"

# def rag_query(question: str, top_k: int = 3):
#     # Detect language
#     lang = "bn" if any('\u0980' <= c <= '\u09FF' for c in question) else "en"

#     # Retrieve from FAISS
#     results = search(question, top_k=top_k)
#     if not results:
#         return {"answer": "❌ কোনো প্রাসঙ্গিক তথ্য পাওয়া যায়নি।", "contexts": [], "language": lang}

#     context = "\n\n".join([r["chunk"] for r in results])

#     # Generate answer
#     answer = generate_answer(context, question, lang)

#     return {
#         "answer": answer,
#         "contexts": [r["chunk"] for r in results],
#         "language": lang
#     }

# # Test
# if __name__ == "__main__":
#     q = "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
#     result = rag_query(q)
#     print("✅ Answer:", result["answer"])



# scripts/4_rag_pipeline.py
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load FAISS index
index = faiss.read_index("../vectorstore/faiss_index.bin")

# Load chunks
with open("../vectorstore/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Load embedding model
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Load LLM (TinyLlama) in 4-bit
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
    load_in_4bit=True
)
llm = pipeline("text-generation", model=model, tokenizer=tokenizer)

def detect_lang(text):
    return "bn" if any('\u0980' <= c <= '\u09FF' for c in text) else "en"

def rag_query(question: str):
    lang = detect_lang(question)
    
    # Embed query
    q_embed = embedding_model.encode([question]).astype('float32')
    
    # Search FAISS
    distances, indices = index.search(q_embed, k=1)
    best_chunk = chunks[indices[0][0]] if indices[0][0] != -1 else "No context found."

    # Build prompt
    if lang == "bn":
        prompt = f"""
        প্রসঙ্গ: {best_chunk}
        প্রশ্ন: {question}
        উত্তর:
        """
    else:
        prompt = f"""
        Context: {best_chunk}
        Question: {question}
        Answer:
        """

    # Generate
    response = llm(prompt, max_new_tokens=100, do_sample=True)
    answer = response[0]['generated_text'][len(prompt):].strip()

    return {
        "answer": answer,
        "context": best_chunk
    }

# Test
if __name__ == "__main__":
    q1 = "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
    q2 = "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?"
    q3 = "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?"

    for q in [q1, q2, q3]:
        result = rag_query(q)
        print(f"\n❓ {q}")
        print(f"✅ {result['answer']}")