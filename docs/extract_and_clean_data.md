# 🗂️ Phase 1: Extract & Clean

## Goal
- Extract text from HSC26 Bangla 1st Paper PDF
- Clean newlines, spaces
- Save as `.txt`
- Insert to MongoDB Atlas

---

1️⃣ **Setup**  
- Used `pytesseract` for OCR.
- Used `PIL` (Python Imaging Library) to open images.
- Connected to MongoDB Atlas using `pymongo` and `.env` config.</br>

2️⃣ **OCR Processing**  
- Read all image files (`.png` / `.jpg`) from `./data/images/` folder.
- Applied Tesseract OCR (`lang="ben"`) for Bangla text.
- Merged all extracted text into one big string. </br>

3️⃣ **Save Raw Text**  
- Saved extracted text locally in `./data/ocr_extracted_text.txt` for inspection.
- Inserted the full text into MongoDB Atlas under `multilinguarag` database and `chunks` collection. </br>

---

## How to Run

1️⃣ PDF put in `data/HSC26-Bangla1st-Paper.pdf` or put images on `data/images/` file </br>
2️⃣ Check `.env` → put Atlas URI  </br>
3️⃣ Activate venv </br>
```bash
source venv/bin/activate
```
4️⃣ Install
```bash
pip install -r requirements.txt
```
5️⃣ Run
```bash
python scripts/extract_and_clean.py
```

---

## ⚙️ Main Libraries

- [`pytesseract`](https://pypi.org/project/pytesseract/): OCR engine.
- [`Pillow`](https://pypi.org/project/Pillow/): Image handling.
- [`pymongo`](https://pypi.org/project/pymongo/): MongoDB connection.
- [`python-dotenv`](https://pypi.org/project/python-dotenv/): Load environment variables.

---

## 🗝️ Important Points

- Make sure `Tesseract` is installed and added to system PATH.
- For Bangla OCR, used `lang="ben"` with trained data.
- MongoDB URI is stored securely in `.env`.

---

## ✅ Output Example

- Total pages processed: (Depends on your images)
- Example book name: `HSC26 Bangla 1st Paper`
- Text length: *(Check `ocr_extracted_text.txt`)*

---

## 📌 Next Step

In **Phase 2**, this raw text is cleaned, noise is removed, and it is chunked for semantic search.

---

