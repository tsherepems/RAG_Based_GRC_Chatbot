"""
streamlit.py
The main entry point for running the Streamlit app.
"""

import streamlit as st
from app import upload_files, manage_files, query_section
from app.chain import VectorStoreManager, RAGChain, QueryProcessor

# Streamlit Page Config
st.set_page_config(
    page_title="GRC RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 1. Initialize Vector Store
if "vector_store" not in st.session_state:
    st.session_state["vector_store_manager"] = VectorStoreManager(
        source_dir="./Data/source",
        persist_dir="./Data/vector_store"
    )
    vector_store = st.session_state["vector_store_manager"].create_or_load_vector_store()
    st.session_state["vector_store"] = vector_store
else:
    vector_store = st.session_state["vector_store"]

# 2. Initialize RAGChain and QueryProcessor
if "rag_chain" not in st.session_state:
    st.session_state["rag_chain"] = RAGChain(vector_store=vector_store)

if "query_processor" not in st.session_state:
    st.session_state["query_processor"] = QueryProcessor(rag_chain=st.session_state["rag_chain"])

# Sidebar
st.sidebar.title("GRC RAG Assistant")
session_name = st.sidebar.text_input("Session Name", value="default_session")

# Tabs (or pages)
tab1, tab2, tab3 = st.tabs(["Upload", "Manage", "Chat"])

with tab1:
    upload_files(session_name)

with tab2:
    manage_files(vector_store)

with tab3:
    query_section(st.session_state["query_processor"])

# Footer
st.markdown("---")
st.markdown("**Powered by a simplified RAG pipeline**")
