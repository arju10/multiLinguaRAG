# Phase 1: Extract & Clean

## Goal
- Extract text from HSC26 Bangla 1st Paper PDF
- Clean newlines, spaces
- Save as `.txt`
- Insert to MongoDB Atlas

---

## How to Run

1️⃣ PDF put in `data/hsc26_bangla_1st_paper.pdf`  
2️⃣ Check `.env` → put Atlas URI  
3️⃣ Activate venv
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