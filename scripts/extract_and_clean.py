import fitz  # PyMuPDF
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from PIL import Image
import pytesseract
import io
import numpy as np

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["multilinguarag"]
collection = db["chunks"]

# Function to detect if text contains Bangla characters
def is_bangla(text: str) -> bool:
    return any('\u0980' <= c <= '\u09FF' for c in text)

# Function to extract text directly from PDF
def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    all_text = ""
    for page in doc:
        all_text += page.get_text() + "\n"
    doc.close()
    return all_text

# Function to convert PDF page to image and extract text via OCR
def extract_text_from_images(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    all_text = ""
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))  # 300 DPI
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        text = pytesseract.image_to_string(img, lang="ben")
        all_text += text + "\n"
    doc.close()
    return all_text

# Main extraction logic
def extract_and_clean(pdf_path: str):
    doc = fitz.open(pdf_path)
    sample_text = ""
    max_sample_pages = min(3, len(doc))
    for page in doc[:max_sample_pages]:
        sample_text += page.get_text() + "\n"
    doc.close()

    if is_bangla(sample_text):
        print("Detected Bangla text. Using OCR-based extraction.")
        all_text = extract_text_from_images(pdf_path)
    else:
        print("Detected English text. Using direct PDF extraction.")
        all_text = extract_text_from_pdf(pdf_path)

    os.makedirs("./data", exist_ok=True)
    output_file = "./data/pdf_extracted_text.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(all_text)
    print(f"Text saved to {output_file}")

    doc_data = {
        "book_name": "HSC26 Bangla 1st Paper",
        "content": all_text,
        "language": "bn" if is_bangla(sample_text) else "en"
    }
    collection.insert_one(doc_data)
    print("Inserted to MongoDB Atlas!")
    return all_text

if __name__ == "__main__":
    pdf_path = "./data/HSC26-Bangla1st-Paper.pdf"
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")
    extract_and_clean(pdf_path)
    client.close()