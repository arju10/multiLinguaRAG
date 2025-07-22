# python scripts/preprocess_utils.py
import unicodedata
import re
import unicodedata
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import regex
import regex as re

load_dotenv()

client = MongoClient(os.getenv("MONGO_URI"))
db = client["multilinguarag"]
collection = db["chunks"]


def normalize_unicode(text: str) -> str:
    # Normalizing Bangla chars
    return unicodedata.normalize('NFKC', text)

def remove_unwanted_chars(text: str) -> str:
    # Keep only Bangla, English, numbers, some punctuations
    pattern = r"[^ঀ-৿a-zA-Z0-9.,?!।:;'\"()\-–—\s]"
    return re.sub(pattern, "", text)

def clean_whitespace(text: str) -> str:
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def normalize_punctuation(text: str) -> str:
    # Multiple full stops --> single
    text = re.sub(r"।+", "।", text)
    text = re.sub(r"\.\.+", ".", text)
    return text

def fix_bangla_ocr_errors(text: str) -> str:
    # Common OCR errors in Bangla
    replacements = {
        'অপরিরিতা': 'অপরিচিতা',
        'অপর্ির্চতা': 'অপরিচিতা',
        'অপরিচিতা': 'অপরিচিতা',  # Normalize variants
        'কলিা।ঘ।': 'ক. ঘ.',
        'কলযাণী': 'কল্যাণী',
        'মামা': 'মামা',
        'অনুপম': 'অনুপম',
        'শুম্ভুনাথ': 'শুম্ভুনাথ',
        'ব্ক': 'ব',       # Fix broken conjuncts
        '্ক': '্',         # Fix misplaced halant
        'ি': 'ি',          # Fix vowel signs
        'ে': 'ে',
        'ো': 'ো',
        'ী': 'ী',
        'ু': 'ু',
        'ূ': 'ূ',
        'ং': 'ং',
        'ঃ': 'ঃ',
        'ঁ': 'ঁ',
        'দ্': 'দ',
        'ন্': 'ন',
        'ম্': 'ম',
        'প্': 'প',
        'ব্': 'ব',
        'ভ্': 'ভ',
        'র্': 'র',
        'ষ্': 'ষ',
        'স্': 'স',
        'ত্': 'ত',
        'ট্': 'ট',
        'ড্': 'ড',
        'ধ্': 'ধ',
        'ল্': 'ল',
        'ক্': 'ক',
        'গ্': 'গ',
        'ঘ্': 'ঘ',
        'চ্': 'চ',
        'ছ্': 'ছ',
        'জ্': 'জ',
        'ঝ্': 'ঝ',
        'ঞ্': 'ঞ',
        'দ্ব': 'দ্ব',
        'ত্র': 'ত্র',
        'জ্ঞ': 'জ্ঞ',
        'ক্ষ': 'ক্ষ',
        'শ্র': 'শ্র',
    }
    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)
    return text

# Remove Page Numbers, Headers, and Noise Lines
def remove_noise_lines(text: str) -> str:
    lines = text.split('\n')
    cleaned_lines = []
    noise_patterns = [
        r'^\s*P[AGE]+\s*\d+',            # PAGE 24
        r'^\s*\?+\s*$',                   # ?????
        r'^\s*SLAns.*$',                  # Answer key lines
        r'^\s*[i|ii|iii|iv]+\s*\.\s*$',   # Bullet points
        r'^\s*উত্তর\s*:',                 # Answer headers
        r'^\s*প্রশ্ন\-\s*\d+',            # "প্রশ্ন- ৩"
        r'^\s*\[\w+\.?\s*দিা\.?\s*\'?\d+', # [ঢা. দিা. '২২]
    ]
    
    for line in lines:
        if len(line.strip()) < 10:  # Too short
            continue
        if any(re.match(pattern, line.strip()) for pattern in noise_patterns):
            continue
        cleaned_lines.append(line.strip())
    
    return '\n'.join(cleaned_lines)

# Fix Broken Words (Conjuncts & Vowel Signs)
def repair_bangla_conjuncts(text: str) -> str:
    # Fix common broken conjuncts
    text = regex.sub(r'([ক-হ])\s+্\s+([ক-হ])', r'\1্য\2', text)  # e.g., ক ্ ষ → ক্ষ
    text = regex.sub(r'([ক-হ])\s+ি', r'\1ি', text)  # fix spacing around vowel signs
    text = regex.sub(r'([ক-হ])\s+ী', r'\1ী', text)
    return text



def remove_garbage_tokens(text: str) -> str:
    # Word tokenize: Bangla, English, Numbers, Punctuation separate
    words = re.findall(r'\w+|[^\s\w]', text, re.UNICODE)

    cleaned_words = []
    for w in words:
        # Only digit or Bengali digit or colon like 1081:39:
        if re.fullmatch(r"[0-9০-৯:]+", w):
            continue
        # Single Bangla letter like ক খ গ ঘ
        if re.fullmatch(r"[অ-ঔক-হ]", w):
            continue
        # Standalone punctuation () বা single dot or single danda
        # if w in {"(", ")", "।", ".", ",", ":", ";", "?", "!", "-", "–", "—"}:
        if w in {"(", ")", "।", ".", ",", ":", ";", "?", "!", "-", "–", "—", "'", '"'}:
            continue
        cleaned_words.append(w)

    return " ".join(cleaned_words)


# Reconstruct Structure: Identify Questions, Answers, Themes
def extract_thematic_sections(text: str) -> dict:
    sections = {
        "questions": [],
        "answers": [],
        "character_analysis": [],
        "themes": []
    }

    # Extract MCQs
    mcq_pattern = r'(\d+।\s*[^।]+।)\s*\[.*?\]\s*\((ক|খ|গ|ঘ)\)\s*উত্তর:\s*(ক|খ|গ|ঘ)'
    for match in regex.finditer(mcq_pattern, text):
        sections["questions"].append(match.group(1))

    # Extract answers
    answer_pattern = r'উত্তর:\s*(ক|খ|গ|ঘ)'
    for match in regex.finditer(answer_pattern, text):
        sections["answers"].append(match.group(1))

    # Extract character analysis
    char_patterns = ['অনুপম', 'কল্যাণী', 'মামা', 'শুম্ভুনাথ', 'মা']
    for char in char_patterns:
        context = regex.findall(rf'(.{{0,100}}{char}[^।]*।{{1,3}}[^।]*।)', text)
        sections["character_analysis"].extend(context)

    # Extract themes
    theme_keywords = ['মাতৃস্নেহ', 'যৌতুক', 'ব্যক্তিত্বহীনতা', 'স্বাধীনতা', 'অপরিচিতা']
    for keyword in theme_keywords:
        context = regex.findall(rf'(.{{0,150}}{keyword}[^।]*।{{1,3}}[^।]*।)', text)
        sections["themes"].extend(context)

    return sections



def semantic_chunking(text: str) -> list:
    chunks = []

    paragraphs = re.split(r'\n\n+', text)
    for para in paragraphs:
        para = para.strip()
        if len(para) < 50:  
            continue

        if len(para) <= 300:  
            chunks.append(para)
        else:
            sentences = re.split(r'[।.!?\n]', para)
            current_chunk = ""
            for sent in sentences:
                sent = sent.strip()
                if len(sent) < 10:
                    continue
                if len(current_chunk) + len(sent) > 300:
                    chunks.append(current_chunk.strip())
                    current_chunk = sent
                else:
                    current_chunk += " " + sent
            if current_chunk:
                chunks.append(current_chunk.strip())

    if len(chunks) <= 1:
        forced_chunks = []
        forced_text = text.strip()
        chunk_size = 300
        for i in range(0, len(forced_text), chunk_size):
            piece = forced_text[i:i+chunk_size].strip()
            if len(piece) > 50:
                forced_chunks.append(piece)
        chunks = forced_chunks

    return chunks





# Final Enhanced Preprocessing Function
def pre_process_enhanced(text: str) -> dict:
    text = normalize_unicode(text)
    text = fix_bangla_ocr_errors(text)
    text = remove_unwanted_chars(text)
    text = repair_bangla_conjuncts(text)
    text = clean_whitespace(text)
    text = normalize_punctuation(text)
    text = remove_noise_lines(text)
    chunks = semantic_chunking(text)
    text = remove_garbage_tokens(text) 

    # Extract structured content
    structured = extract_thematic_sections(text)

    # Create semantic chunks

    return {
        "cleaned_text": text,
        "chunks": chunks,
        "structured": structured
    }


# --- Run directly ---
doc = collection.find_one({"book_name": "HSC26 Bangla 1st Paper"})
raw = doc["content"]
print(f"Raw Length: {len(raw)}")

result = pre_process_enhanced(raw)
cleaned = result["cleaned_text"]
chunks = result["chunks"]
sections = result["structured"]

print(f"Cleaned Length: {len(cleaned)} | Chunks: {len(chunks)}")

# Save to MongoDB
collection.update_one(
    {"_id": doc["_id"]},
    {"$set": {
        "cleaned_content": cleaned,
        "semantic_chunks": chunks,
        "sections": sections
    }}
)
print("✅ Saved cleaned, chunks & sections to MongoDB!")

# Save locally
os.makedirs("./data", exist_ok=True)
with open("./data/cleaned_text_HSC26.txt", "w", encoding="utf-8") as f:
    f.write(cleaned)
with open("./data/chunks_HSC26.txt", "w", encoding="utf-8") as f:
    for chunk in chunks:
        f.write(chunk + "\n---\n")
print("✅ Saved cleaned & chunks locally!")
