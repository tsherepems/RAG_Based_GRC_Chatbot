# RAG_Based_GRC_Chatbot
A Retrieval-Augmented Generation (RAG) system that allows users to upload documents, manage files, and query the contents using an interactive Streamlit interface. This project integrates file handling, document chunking, vector storage, and query processing powered by Google’s Generative AI and LangChain libraries.

## Overview

The RAG Knowledge Assistant provides a modular system with a clear separation of UI and business logic. Users can:

- **Upload Documents:** Support for PDF, DOCX, and TXT formats.
- **Manage Files:** Delete or update uploaded documents.
- **Query Content:** Ask questions that are processed through a pipeline that validates, refines, retrieves, and generates answers using LLMs.

The backend utilizes vector stores (via Chroma) and text chunking (via LangChain’s RecursiveCharacterTextSplitter) to process and index document contents, while the frontend is built entirely in Streamlit.

## Project Structure

- **streamlit.py:**  
  The main entry point for the application. Handles UI rendering, session state management, and interaction with backend services.  
  **Reference:** `streamlit`

- **core.py:**  
  Contains functions for file operations (saving, loading, parsing) and text chunking.  
  **Reference:** `core`

- **chain.py:**  
  Implements the RAG pipeline, including:
  - Query validation (with Perspective API)
  - Query refinement
  - Document re-ranking
  - Vector store management
  - Query processing and answer generation  
  **Reference:** `chain`

- **interface.py:**  
  Provides Streamlit UI adapter functions for uploading files, managing documents, and handling query input/output.  
  **Reference:** `interface`

- **__init__.py:**  
  Re-exports essential functions and classes for ease of use.  
  **Reference:** `__init__`

- **requirements.txt:**  
  Lists the dependencies required for the project, including LangChain, Streamlit, python-docx, pymupdf4llm, and more.  
  **Reference:** `requirements.txt`
