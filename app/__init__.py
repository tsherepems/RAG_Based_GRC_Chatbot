"""
app/__init__.py
Simple init file for the 'app' package.
"""

# You can re-export classes and functions here if you like, or leave it minimal.
from .core import save_uploaded_file, load_file, load_existing_files, chunk_documents
from .interface import upload_files, manage_files, query_section
from .chain import QueryProcessor, VectorStoreManager, RAGChain, QueryRefiner, QueryValidator

