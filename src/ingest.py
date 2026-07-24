import os
from pypdf import PdfReader
from pdf2image import convert_from_path
import pytesseract

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Minimum characters we'd expect from a real typed page. Below this,
# we assume the page has no text layer (i.e. it's a scanned image) and OCR it.
MIN_CHARS_PER_PAGE = 20


def load_txt_or_md(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def ocr_page(filepath: str, page_number: int) -> str:
    """Renders one PDF page as an image and runs OCR on it."""
    images = convert_from_path(filepath, first_page=page_number, last_page=page_number)
    return pytesseract.image_to_string(images[0])


def load_pdf(filepath: str) -> str:
    reader = PdfReader(filepath)
    text = ""
    for i, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if len(page_text.strip()) < MIN_CHARS_PER_PAGE:
            page_text = ocr_page(filepath, i)
        text += page_text + "\n"
    return text


def load_all_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """
    Returns a list of {"source": filename, "text": full_text}
    one entry per file in data_dir.
    """
    documents = []
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        if filename.endswith((".md", ".txt")):
            text = load_txt_or_md(filepath)
        elif filename.endswith(".pdf"):
            text = load_pdf(filepath)
        else:
            continue
        documents.append({"source": filename, "text": text})
    return documents


if __name__ == "__main__":
    docs = load_all_documents()
    print(f"Loaded {len(docs)} document(s):")
    for d in docs:
        print(f"  - {d['source']}  ({len(d['text'])} characters)")
