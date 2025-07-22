
# import pdfplumber

# # Path to your PDF file
# pdf_path = "./data/HSC26-Bangla1st-Paper-1.pdf"

# # Open the PDF and extract text
# with pdfplumber.open(pdf_path) as pdf:
#     for page in pdf.pages:
#         text = page.extract_text()
#         if text:
#             print(text)
#             # Optionally, save to a file
#             with open("extracted_bangla_text.txt", "a", encoding="utf-8") as f:
#                 f.write(text + "\n")


# import fitz  # PyMuPDF

# pdf_path = "./data/HSC26-Bangla1st-Paper.pdf"
# doc = fitz.open(pdf_path)
# for page in doc:
#     text = page.get_text("text")
#     if text:
#         with open("./data/extracted_bangla_text.txt", "a", encoding="utf-8") as f:
#             f.write(text + "\n")
