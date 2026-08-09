import fitz
from docx import Document
import json
import csv
from pathlib import Path


def load_pdf(file):
    """Extract text from PDF."""

    text = ""

    doc = fitz.open(
        stream=file.getvalue(),
        filetype="pdf"
    )

    for page in doc:
        text += page.get_text()

    doc.close()

    return text


def load_docx(path):
    """Extract text from DOCX."""

    doc = Document(path)

    return "\n".join(
        para.text
        for para in doc.paragraphs
    )


def load_text(file):
    """Extract text from TXT/MD."""

    return file.getvalue().decode(
        "utf-8",
        errors="ignore"
    )


def load_json(file):
    """Extract JSON as formatted text."""

    try:
        data = json.loads(
            file.getvalue().decode("utf-8")
        )

        return json.dumps(
            data,
            indent=2
        )

    except Exception:
        return ""


def load_csv(file):
    """Extract CSV rows."""

    try:

        text = file.getvalue().decode(
            "utf-8",
            errors="ignore"
        )

        rows = []

        reader = csv.reader(text.splitlines())

        for row in reader:
            rows.append(", ".join(row))

        return "\n".join(rows)

    except Exception:
        return ""


def extract_text(file, save_path):
    """
    Automatically detect file type
    and return extracted text.
    """

    suffix = Path(file.name).suffix.lower()

    if suffix == ".pdf":
        return load_pdf(file)

    elif suffix == ".docx":
        return load_docx(save_path)

    elif suffix in [".txt", ".md"]:
        return load_text(file)

    elif suffix == ".json":
        return load_json(file)

    elif suffix == ".csv":
        return load_csv(file)

    return ""