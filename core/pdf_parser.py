import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz  # PyMuPDF


def parse_pdf(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")

    doc = fitz.open(file_path)
    full_text = []
    page_count = len(doc)  # pehle save karo

    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        lines = text.split('\n')
        cleaned = []
        for line in lines:
            line = line.strip()
            if not line or line.isdigit() or len(line) < 3:
                continue
            cleaned.append(line)

        page_text = ' '.join(cleaned)
        if page_text:
            full_text.append(page_text)

    doc.close()

    final_text = ' '.join(full_text)
    print(f"[PDF Parser] Extracted {len(final_text)} chars from {page_count} pages")
    return final_text


def parse_pdf_bytes(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = []
    page_count = len(doc)  # pehle save karo

    for page in doc:
        text = page.get_text("text")
        lines = text.split('\n')
        cleaned = []
        for line in lines:
            line = line.strip()
            if not line or line.isdigit() or len(line) < 3:
                continue
            cleaned.append(line)

        page_text = ' '.join(cleaned)
        if page_text:
            full_text.append(page_text)

    doc.close()

    final_text = ' '.join(full_text)
    print(f"[PDF Parser] Extracted {len(final_text)} chars from {page_count} pages")
    return final_text