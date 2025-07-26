from .query_engine import search
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

model_id = "Qwen/Qwen2-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
use_cuda = torch.cuda.is_available()

if use_cuda:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=True,
        low_cpu_mem_usage=True
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="auto"
    )

llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    temperature=0.4,
    top_p=0.9,
    do_sample=True,
)

conversation_history = []

def generate_answer(context: str, question: str, lang: str = "bn"):
    if lang == "bn":
        prompt = f"""<প্রসঙ্গ>
{context}
</প্রসঙ্গ>

<প্রশ্ন>
{question}
</প্রশ্ন>

<নির্দেশ>
প্রসঙ্গের তথ্যের ভিত্তিতে প্রশ্নের উত্তর দিন। উত্তরটি সংক্ষিপ্ত এবং সুনির্দিষ্ট হতে হবে। যদি প্রসঙ্গে উত্তর না থাকে, তবে বলুন "প্রসঙ্গে উত্তর পাওয়া যায়নি।" কোনো বাহ্যিক তথ্য যোগ করবেন না।
</নির্দেশ>

<উত্তর>
"""
    else:
        prompt = f"""<Context>
{context}
</Context>

<Question>
{question}
</Question>

<Instructions>
Answer the question based solely on the context provided. Keep the answer concise and specific. If the answer is not found in the context, state "Answer not found in context." Do not add external information.
</Instructions>

<Answer>
"""
    sequences = llm_pipeline(prompt)
    return sequences[0]['generated_text'][len(prompt):].strip()

def rag_query(question: str, top_k: int = 5):  # Increased top_k for better context
    lang = "bn" if any('\u0980' <= c <= '\u09FF' for c in question) else "en"
    results = search(question, top_k=top_k, similarity_threshold=0.6)  # Raised threshold
    context = "\n\n".join([r["chunk"] for r in results])
    history_context = "\n".join([f"Q: {q} A: {a}" for q, a in conversation_history[-3:]] if conversation_history else [])
    full_context = f"{history_context}\n\n{context}" if history_context else context

    try:
        answer = generate_answer(full_context, question, lang)
        conversation_history.append((question, answer))
    except Exception as e:
        print(f"Error generating answer: {e}")
        answer = "প্রসঙ্গে উত্তর পাওয়া যায়নি।" if lang == "bn" else "Answer not found in context."

    return {
        "answer": answer,
        "contexts": [r["chunk"] for r in results],
        "language": lang
    }

if __name__ == "__main__":
    test_questions = [
        "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
        "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?",
        "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?"
    ]
    for q in test_questions:
        result = rag_query(q)
        print(f"Question: {q}")
        print(f"Answer: {result['answer']}")
        print(f"Contexts: {result['contexts'][:2]}")  # Print first 2 contexts for brevity
        print("-" * 50)