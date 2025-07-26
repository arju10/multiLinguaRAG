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
    pattern = r"[^ঀ-৿a-zA-Z0-9.,?!।|:;'\"()\-–—\s]"
    return re.sub(pattern, "", text)

def clean_whitespace(text: str) -> str:
    text = re.sub(r"\s*\|\s*", " | ", text)
    text = text.replace("\t", " ")
    text = re.sub(r" +", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n+", "\n", text)

    return text.strip()

def normalize_punctuation(text: str) -> str:
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
        'শসতুনাথ': 'শুম্ভুনাথ',
        'ছাবিবিশ':'ছাব্বিশ',
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
        r'^\s*PAGE\s*\d+',                  # PAGE 24 বা PAGE24
        r'^\s*\?+\s*$',                    # শুধু প্রশ্নচিহ্ন
        r'^\s*SLAns.*$',                   # SLAns দিয়ে শুরু
        r'^\s*(i|ii|iii|iv)+\s*\.\s*$',   # Roman numerals bullets
        r'^\s*উত্তর\s*:',                 # উত্তর:
        r'^\s*প্রশ্ন\-\s*\d+',             # প্রশ্ন- ৩
        r'^\s*\[\w+\.?\s*দিা\.?\s*\'?\d+', # [ঢা. দিা. '২২]
    ]

    for line in lines:
        line = line.strip()
        if len(line) < 10:
            continue
        if any(re.match(pattern, line) for pattern in noise_patterns):
            continue
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


# Fix Broken Words (Conjuncts & Vowel Signs)
def repair_bangla_conjuncts(text: str) -> str:
    # Fix common broken conjuncts
    text = regex.sub(r'([ক-হ])\s*্\s*([ক-হ])', r'\1্\2', text)
    # Fix spacing around vowel signs
    text = regex.sub(r'([ক-হ])\s+ি', r'\1ি', text)
    text = regex.sub(r'([ক-হ])\s+ী', r'\1ী', text)

    return text

def remove_garbage_tokens(text: str) -> str:
    words = re.findall(r'\w+|[^\s\w]', text, re.UNICODE)
    cleaned_words = []
    for w in words:
        # Ignore pure numbers (বাংলা/ইংরেজি), একক অক্ষর, এবং punctuation (যা প্রয়োজন)
        if re.fullmatch(r"[0-9০-৯]+", w):
            continue
        if len(w) == 1 and re.fullmatch(r"[অ-ঔক-হ]", w):
            continue
        if w in {"(", ")", ".", ",", ":", ";", "?", "!", "-", "–", "—", "'", '"'}:
            continue
        cleaned_words.append(w)
    return " ".join(cleaned_words)


def remove_small_numbers(text: str) -> str:
    return re.sub(r'\b[০-৯0-9]{1,8}\b', '', text)

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


import re

def semantic_chunking(text: str, max_chunk_size=300, min_chunk_size=50):
    chunks = []
    paragraphs = re.split(r'\n\n+', text)
    for para in paragraphs:
        para = para.strip()
        if len(para) < min_chunk_size:
            continue
        if len(para) <= max_chunk_size:
            chunks.append(para)
        else:
            sentences = re.split(r'[।.!?]', para)
            current = ""
            for sent in sentences:
                sent = sent.strip()
                if len(sent) < 10:
                    continue
                if len(current) + len(sent) + 1 > max_chunk_size:
                    chunks.append(current.strip())
                    current = sent
                else:
                    current += " " + sent
            if current:
                chunks.append(current.strip())
    if len(chunks) <= 2:
        chunks = [text[i:i+max_chunk_size].strip() for i in range(0, len(text), max_chunk_size)]
    return chunks




# Final Enhanced Preprocessing Function
def pre_process_enhanced(text: str) -> dict:
    text = normalize_unicode(text)
    # text = clean_whitespace(text)
    # text = fix_bangla_ocr_errors(text) #(Problemetic)
    text = remove_unwanted_chars(text)
    # text = repair_bangla_conjuncts(text)
    text = normalize_punctuation(text)
    text = remove_small_numbers (text)
    # text = remove_noise_lines(text)
    # text = remove_garbage_tokens(text) 

    chunks = semantic_chunking(text)

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
        f.write(chunk + "\n")

print("✅ Saved cleaned & chunks locally without --- !")
