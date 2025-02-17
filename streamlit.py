import streamlit as st

# from app.core import load_file

from app import (
    upload_files,
    manage_files,
    query_section,
    QueryProcessor,
    VectorStoreManager,
    RAGChainProcessor,
    QueryRefiner,
    DocumentReranker,
    
)
# from app.chain import QueryRefiner, DocumentReranker, RAGChainProcessor, QueryProcessor, VectorStoreManager
from config import api_key
from langchain_google_genai import ChatGoogleGenerativeAI
import time

# Page Configuration
st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stButton > button { width: 100%; border-radius: 5px; height: 3em; background-color: #FF4B4B; color: white; }
    .stTextInput > div > div > input { border-radius: 5px; }
    .success-message { padding: 1rem; border-radius: 5px; background-color: #d4edda; color: #155724; margin-bottom: 1rem; }
    .error-message { padding: 1rem; border-radius: 5px; background-color: #f8d7da; color: #721c24; margin-bottom: 1rem; }
    .sidebar .sidebar-content { background-color: #f8f9fa; }
    .status-indicator { height: 10px; width: 10px; border-radius: 50%; display: inline-block; margin-right: 5px; }
    .status-active { background-color: #28a745; }
    .status-inactive { background-color: #dc3545; }
    </style>
    """, unsafe_allow_html=True)

# Initialize Vector Store and Components
if "vector_store" not in st.session_state:
    vector_store_manager = VectorStoreManager(
        source_dir="./Data/source",
        persist_dir="./Data/vector_store",
        api_key=api_key
    )
    st.session_state["vector_store"] = vector_store_manager.create_vector_store()
    st.session_state["chat_history"] = []
    st.session_state["system_status"] = "active"

vector_store = st.session_state["vector_store"]


# Initialize QueryProcessor components
query_refiner = QueryRefiner(api_key)
document_reranker = DocumentReranker(api_key)
rag_chain_processor = RAGChainProcessor(vector_store, api_key)

# Initialize QueryProcessor with required components
query_processor = QueryProcessor(
    vector_store_manager=vector_store_manager,
    rag_pipeline=rag_chain_processor,
    query_refiner=query_refiner,
    #document_reranker=document_reranker, 
    reranker=document_reranker,
    llm=ChatGoogleGenerativeAI(
        google_api_key=api_key,
        model="gemini-1.5-flash",
        temperature=0.5,
    ),
)
# Sidebar
with st.sidebar:
    st.image("./Data/img/img-masthead-Security-Audit.png", width=200)  # Logo
    st.markdown("---")

    # System Status
    status_color = "status-active" if st.session_state["system_status"] == "active" else "status-inactive"
    st.markdown(f"""
        <div>
            <span class='status-indicator {status_color}'></span>
            <span>System Status: {st.session_state["system_status"].title()}</span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Session Management
    st.subheader("Session Management")
    session_name = st.text_input(
        "Session Name",
        value="default",
        help="Enter a unique name for your upload session"
    )

    # Clear Session
    if st.button("Clear Session"):
        st.session_state["chat_history"] = []
        st.success("Session cleared successfully!")

# Main Content
st.title("🤖 RAG Knowledge Assistant")

# Tabs for different sections
tab1, tab2, tab3 = st.tabs(["Upload Files", "Manage Documents", "Chat Interface"])

with tab1:
    st.header("Document Upload")
    col1, col2 = st.columns([2, 1])

    with col1:
        try:
            upload_files(session_name)
        except Exception as e:
            st.error(f"Error during upload: {str(e)}")

    with col2:
        st.info("""
        📁 Supported file formats:
        - PDF (.pdf)
        - Word (.docx)
        - Text (.txt)
        """)

with tab2:
    st.header("Document Management")
    try:
        manage_files(vector_store)
    except Exception as e:
        st.error(f"Error managing files: {str(e)}")

# Use QueryProcessor in your chat section
with tab3:
    st.header("Knowledge Assistant")

    # Chat Interface
    for message in st.session_state.get("chat_history", []):
        role = message["role"]
        content = message["content"]

        if role == "user":
            st.markdown(f"**You:** {content}")
        else:
            st.markdown(f"**Assistant:** {content}")

    # Query Input
    query = st.text_input("Ask a question about your documents or a general question:", key="query_input")

    if st.button("Send", key="send_button"):
        if query:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": query})

            # Show typing indicator
            with st.spinner("Thinking..."):
                try:
                    response = query_processor.process_query(query)
                    time.sleep(0.5)  # Small delay for better UX

                    # Add assistant response to chat history
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": response}
                    )

                    # Force rerun to update chat history
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
        else:
            st.warning("Please enter a question.")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("Made with ❤️ by Your Company")
with col2:
    st.markdown("Version 1.0.0")
with col3:
    st.markdown("[Documentation](https://your-docs-link.com)")
