"""
app/interface.py
Provides Streamlit interface functions for file upload, management, and query input.
"""

import streamlit as st
import os
from pathlib import Path

from app.core import save_uploaded_file
from langchain_chroma import Chroma

DATA_DIR = "./Data/source"

def upload_files(session_name: str) -> None:
    """
    Allows users to upload files and saves them into a session-specific directory.
    """
    st.subheader("Upload Files")
    uploaded_files = st.file_uploader(
        "Upload your files (.pdf, .docx, .txt)",
        accept_multiple_files=True
    )

    if uploaded_files:
        session_dir = f"{DATA_DIR}/{session_name}" if session_name else DATA_DIR
        os.makedirs(session_dir, exist_ok=True)

        for file in uploaded_files:
            save_uploaded_file(file, session_dir)
            st.success(f"File '{file.name}' saved to '{session_dir}'!")
        st.info("Files uploaded successfully.")

def manage_files(vector_store: Chroma) -> None:
    """
    Lists all files in the source directory and allows deleting them.
    """
    st.subheader("Manage Uploaded Files")
    uploaded_files = list(Path(DATA_DIR).rglob("*"))

    if uploaded_files:
        file_names = [f.relative_to(DATA_DIR) for f in uploaded_files if f.is_file()]
        selected_file = st.selectbox("Select a file to delete:", file_names)

        if st.button("Delete Selected File"):
            file_path = Path(DATA_DIR) / selected_file
            if file_path.exists():
                file_path.unlink()
                st.success(f"Deleted file: {selected_file}")
            else:
                st.error("File not found.")
    else:
        st.info("No files have been uploaded yet.")

def query_section(query_processor) -> None:
    """
    Renders a simple query input box and displays the answer.
    """
    st.subheader("Ask Your Question")
    user_query = st.text_input("Enter your question here:")

    if st.button("Submit Query"):
        if user_query.strip():
            with st.spinner("Processing..."):
                response = query_processor.process_query(user_query)
                st.write("**Answer:**")
                st.write(response)
        else:
            st.warning("Please enter a valid query.")
