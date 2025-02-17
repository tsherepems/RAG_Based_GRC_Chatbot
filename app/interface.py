
import streamlit as st
from pathlib import Path
from app.core import save_uploaded_file
from langchain_chroma import Chroma
import os

# Constants
DATA_DIR = "./Data/source"
os.makedirs(DATA_DIR, exist_ok=True)  # Ensure the data directory exists

def upload_files(session_name: str) -> None:
    """
    Handles file uploads and saves them into session-specific subdirectories.
    """
    st.header("Upload Files")
    uploaded_files = st.file_uploader("Upload your files (.pdf, .docx, .txt)", accept_multiple_files=True)

    if uploaded_files:
        session_dir = f"{DATA_DIR}/{session_name}" if session_name else DATA_DIR
        os.makedirs(session_dir, exist_ok=True)

        for file in uploaded_files:
            save_uploaded_file(file, session_dir)
            st.success(f"File '{file.name}' saved to '{session_dir}'!")

        st.info("Files uploaded successfully.")

def manage_files(vector_store: Chroma) -> None:
    """
    Displays uploaded files and allows users to delete them dynamically.
    """
    st.header("Manage Uploaded Files")
    uploaded_files = list(Path(DATA_DIR).rglob("*"))

    if uploaded_files:
        st.write("Uploaded Files:")
        file_to_delete = st.selectbox("Select a file to delete:", [file.name for file in uploaded_files])

        if st.button("Delete Selected File"):
            file_path = Path(DATA_DIR) / file_to_delete
            if file_path.exists():
                file_path.unlink()  # Delete the file
                st.success(f"File '{file_to_delete}' has been deleted!")

                # Reload the vector store after deletion
                vector_store = Chroma(persist_directory="./Data/output")
                st.session_state["vector_store"] = vector_store
            else:
                st.error(f"File '{file_to_delete}' does not exist.")
    else:
        st.info("No files uploaded yet.")

def query_section(query_processor) -> None:
    """
    Handles user query input and displays the generated response.
    """
    st.header("Ask Your Question")
    query = st.text_input("Enter your question:")

    if query:
        st.write(f"**Query:** {query}")
        with st.spinner("Processing..."):
            try:
                response = query_processor.process_query(query)
                st.subheader("Response")
                st.write(response)
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
