import pytesseract
from PIL import Image
import os

from pymongo import MongoClient
from dotenv import load_dotenv

# Load env
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
# print("MongoUri: ", MONGO_URI)
client = MongoClient(MONGO_URI)
db = client["multilinguarag"]
collection = db["chunks"]

# Folder-> Data PNG/JPGs are stored
image_folder = "./data/images/"

all_text = ""

# Process each image
for img_file in sorted(os.listdir(image_folder)):
    if img_file.endswith(".png") or img_file.endswith(".jpg"):
        img_path = os.path.join(image_folder, img_file)
        img = Image.open(img_path)

        # OCR
        text = pytesseract.image_to_string(img, lang="ben")
        all_text += text + "\n"

print("OCR Complete!")

# Save to file for inspection
with open("./data/ocr_extracted_text.txt", "w", encoding="utf-8") as f:
    f.write(all_text)

print("Text saved to ocr_extracted_text.txt")

# Insert to MongoDB Atlas
doc_data = {
    "book_name": "HSC26 Bangla 1st Paper",
    "content": all_text
}

collection.insert_one(doc_data)
print("Inserted to MongoDB Atlas!")

client.close()
