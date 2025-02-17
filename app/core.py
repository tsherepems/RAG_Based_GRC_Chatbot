
import os
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from docx import Document as DocxDocument
from typing import List
import pymupdf4llm
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = "./Data/source"


def save_uploaded_file(file, upload_session_dir):
    """
    Saves an uploaded file into a session-specific subdirectory under the data directory.
    """
    os.makedirs(upload_session_dir, exist_ok=True)
    save_path = Path(upload_session_dir) / file.name
    with open(save_path, "wb") as f:
        f.write(file.read())
    logger.info(f"File saved to {save_path}")
    return save_path

def load_existing_files(source_dir: str = DATA_DIR) -> List[Document]:
    """
    Loads all existing files in the source_dir and returns them as a list of LangChain Documents.
    Each Document has:
      - page_content: The textual content of the file
      - metadata:     A dict with both "source" (full path) and "file_name" (basename)
    """
    documents = []
    for file_path in Path(source_dir).rglob("*"):
        if file_path.is_file():
            try:
                content = load_file(file_path)
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": str(file_path),
                        "file_name": file_path.name,  # Store just the filename
                    }
                )
                documents.append(doc)
            except ValueError as e:
                logger.warning(f"Skipping unsupported file: {file_path} ({e})")
    logger.info(f"Loaded {len(documents)} documents from {source_dir}")
    return documents


def load_file(file_path: Path) -> str:
    """
    Loads content from a file based on its extension (.txt, .docx, .pdf).
    Throws ValueError if format is unsupported.
    """
    ext = file_path.suffix.lower()

    if ext == ".txt":
        return _load_text_file(file_path)
    elif ext == ".docx":
        return _load_docx_file(file_path)
    elif ext == ".pdf":
        return _load_pdf_file(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def _load_text_file(file_path: Path) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        logger.error(f"Error parsing TXT file {file_path}: {e}")
        return ""


def _load_docx_file(file_path: Path) -> str:
    try:
        doc = DocxDocument(file_path)
        return "\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip())
    except Exception as e:
        logger.error(f"Error parsing DOCX file {file_path}: {e}")
        return ""


def _load_pdf_file(file_path: Path) -> str:
    try:
        return pymupdf4llm.to_markdown(str(file_path))
    except Exception as e:
        logger.error(f"Error parsing PDF with pymupdf4llm: {e}")
        return ""


def chunk_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Document]:
    """
    Splits documents into smaller chunks using RecursiveCharacterTextSplitter.
    Returns a list of new Document chunks (each chunk has the same metadata as the original).
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    try:
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Generated {len(chunks)} chunks from {len(documents)} original documents.")
        return chunks
    except Exception as e:
        logger.error(f"Error during document chunking: {e}")
        return []

# import os
# from pathlib import Path
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.schema import Document
# from docx import Document as DocxDocument
# import pymupdf4llm

# DATA_DIR = "./Data/source"


# def save_uploaded_file(file, upload_session_dir):
#     """
#     Saves an uploaded file into a session-specific subdirectory under the data directory.
#     """
#     os.makedirs(upload_session_dir, exist_ok=True)
#     save_path = Path(upload_session_dir) / file.name
#     with open(save_path, "wb") as f:
#         f.write(file.read())
#     return save_path


# def load_existing_files(source_dir=DATA_DIR):
#     """
#     Loads all existing files in the source directory and returns them as a list of Documents.
#     """
#     documents = []
#     for file_path in Path(source_dir).rglob("*"):  # Recursively load all files
#         if file_path.is_file():
#             content = load_file(file_path)
#             documents.append(Document(page_content=content, metadata={"source": str(file_path)}))
#     return documents


# def load_file(file_path: str) -> str:
#     """
#     Loads content from a file based on its extension (.txt, .docx, .pdf).
#     """
#     ext = Path(file_path).suffix.lower()

#     if ext == ".txt":
#         with open(file_path, "r", encoding="utf-8") as file:
#             return file.read()
#     elif ext == ".docx":
#         try:
#             doc = DocxDocument(file_path)
#             return "\n".join(paragraph.text for paragraph in doc.paragraphs)
#         except Exception as e:
#             print(f"Error parsing DOCX file {file_path}: {e}")
#             return ""
#     elif ext == ".pdf":
#         try:
#             return pymupdf4llm.to_markdown(str(file_path))
#         except Exception as e:
#             print(f"Error parsing PDF with pymupdf4llm: {e}")
#             return ""
#     else:
#         raise ValueError(f"Unsupported file format: {ext}")


# def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
#     """
#     Splits documents into manageable chunks using RecursiveCharacterTextSplitter.
#     """
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#     return text_splitter.split_documents(documents)
