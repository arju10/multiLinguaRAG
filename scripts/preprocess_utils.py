import unicodedata
import re
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
client = MongoClient(os.getenv("MONGO_URI"))
db = client["multilinguarag"]
collection = db["chunks"]

# Unicode Normalization
def normalize_unicode(text: str) -> str:
    return unicodedata.normalize('NFKC', text)

# Remove Unwanted Characters
def remove_unwanted_chars(text: str) -> str:
    pattern = r"[^ঀ-৿a-zA-Z0-9.,?!।|:;'\"()\-–—\s]"
    return re.sub(pattern, "", text)

# Clean Whitespace
def clean_whitespace(text: str) -> str:
    text = re.sub(r"\s*\|\s*", " | ", text)
    text = text.replace("\t", " ")
    text = re.sub(r" +", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()

# Normalize Punctuation
def normalize_punctuation(text: str) -> str:
    text = re.sub(r"।+", "।", text)
    text = re.sub(r"\.\.+", ".", text)
    return text

# Remove Noise Lines
def remove_noise_lines(text: str) -> str:
    lines = text.split('\n')
    cleaned_lines = []
    noise_patterns = [
        r'^\s*PAGE\s*\d+',
        r'^\s*\?+\s*$',
        r'^\s*SLAns.*$',
        r'^\s*(i|ii|iii|iv)+\s*\.\s*$',
        r'^\s*উত্তর\s*:',
        r'^\s*প্রশ্ন\-\s*\d+',
        r'^\s*\[\w+\.?\s*দিা\.?\s*\'?\d+',
    ]
    for line in lines:
        line = line.strip()
        if len(line) < 10:
            continue
        if any(re.match(pattern, line) for pattern in noise_patterns):
            continue
        cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)

# Hybrid Chunking
def advanced_bangla_chunking(text: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", "। ", "।", "? ", "!", ". ", " ", ""],
        length_function=len,
        keep_separator=True
    )
    chunks = splitter.split_text(text)
    return [c.strip() for c in chunks if len(c.strip()) > 50]

# Final Preprocessing
def pre_process_enhanced(text: str) -> dict:
    text = normalize_unicode(text)
    text = remove_unwanted_chars(text)
    text = remove_noise_lines(text)
    text = normalize_punctuation(text)
    text = clean_whitespace(text)
    chunks = advanced_bangla_chunking(text)
    return {
        "cleaned_text": text,
        "chunks": chunks
    }

if __name__ == "__main__":
    doc = collection.find_one({"book_name": "HSC26 Bangla 1st Paper"})
    if not doc:
        raise ValueError("❌ No document found in MongoDB! Run preprocessing first.")

    raw = doc["content"]
    print(f"Raw Length: {len(raw)}")

    result = pre_process_enhanced(raw)
    cleaned = result["cleaned_text"]
    chunks = result["chunks"]

    print(f"Cleaned Length: {len(cleaned)} | Chunks: {len(chunks)}")
    print("Sample Chunks:")
    for i in range(min(5, len(chunks))):
        print(f"{i+1}: {chunks[i][:200]}...")

    collection.update_one(
        {"_id": doc["_id"]},
        {"$set": {
            "cleaned_content": cleaned,
            "semantic_chunks": chunks
        }}
    )
    print("✅ Saved cleaned & chunks to MongoDB!")

    os.makedirs("./data", exist_ok=True)
    with open("./data/cleaned_text_HSC26.txt", "w", encoding="utf-8") as f:
        f.write(cleaned)
    with open("./data/chunks_HSC26.txt", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk + "\n")
    print("✅ Saved cleaned & chunks locally!")