from pathlib import Path
from collections import Counter


DATA_FOLDER = Path("data")
VECTORSTORE_FOLDER = Path("vectorstore")


def get_document_stats():
    """Return overall document statistics."""

    files = list(DATA_FOLDER.glob("*"))

    total_documents = len(files)

    total_size = sum(
        file.stat().st_size
        for file in files
        if file.is_file()
    )

    total_size_mb = round(total_size / (1024 * 1024), 2)

    file_types = Counter(
        file.suffix.lower().replace(".", "").upper()
        for file in files
        if file.is_file()
    )

    recent_upload = (
        max(files, key=lambda x: x.stat().st_mtime).name
        if files
        else "None"
    )

    return {
        "documents": total_documents,
        "storage_mb": total_size_mb,
        "file_types": dict(file_types),
        "recent_upload": recent_upload,
    }


def get_total_chunks():
    """
    Estimate whether vectors have been created.
    Exact chunk count isn't stored by FAISS,
    so this simply reports availability.
    """

    index_file = VECTORSTORE_FOLDER / "index.faiss"

    if index_file.exists():
        return "Indexed"

    return "Not Indexed"


def get_file_type_chart_data():

    stats = get_document_stats()

    labels = list(stats["file_types"].keys())
    values = list(stats["file_types"].values())

    return labels, values