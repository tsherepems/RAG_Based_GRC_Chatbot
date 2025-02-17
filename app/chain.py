from langchain_chroma import Chroma
from app.core import load_existing_files, chunk_documents
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from pathlib import Path
from typing import List, Dict, Any
import logging
import os
import re
import json
import requests

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Perspective API for query validation
from config import api_key, Perspective_api


### QueryValidator Class ###
class QueryValidator:
    """Validates user queries, including checks for offensive content using the Perspective API."""

    def validate_query(self, query: str) -> Dict[str, Any]:
        """
        Validate query for offensive content, generic queries, and formatting issues.
        Uses the Perspective API to check for toxicity.
        """
        # Basic validation
        if not query.strip():
            return {"valid": False, "reason": "Empty query", "is_generic": False}
        if len(query) > 500:
            return {"valid": False, "reason": "Query too long", "is_generic": False}
        if re.search(r"[<>$;]", query):
            return {"valid": False, "reason": "Invalid characters detected", "is_generic": False}

        # Detect generic queries
        generic_patterns = [r"^(hi|hello|hey)[\s!]*$", r"^how are you", r"^what('s| is) up"]
        is_generic = any(re.match(pattern, query.lower()) for pattern in generic_patterns)

        # Perspective API call
        perspective_url = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"
        headers = {"Content-Type": "application/json"}
        data = {
            "comment": {"text": query},
            "languages": ["en"],
            "requestedAttributes": {"TOXICITY": {}, "PROFANITY": {}},
            "doNotStore": True,
        }
        try:
            response = requests.post(
                perspective_url, headers=headers, json=data, params={"key": Perspective_api}
            )
            response_data = response.json()

            toxicity_score = response_data.get("attributeScores", {}).get("TOXICITY", {}).get("summaryScore", {}).get("value", 0)
            profanity_score = response_data.get("attributeScores", {}).get("PROFANITY", {}).get("summaryScore", {}).get("value", 0)

            if toxicity_score >= 0.7 or profanity_score >= 0.7:
                return {
                    "valid": False,
                    "reason": "Offensive or inappropriate content detected",
                    "is_generic": is_generic,
                }
        except Exception as e:
            logger.warning(f"Perspective API error: {str(e)}")

        return {"valid": True, "reason": "", "is_generic": is_generic}


### QueryRefiner Class ###
class QueryRefiner:
    """Refines user queries using an LLM."""

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=api_key, model=model, temperature=0.1, max_tokens=100
        )

    def refine_query(self, query: str, doc_context: str = "") -> str:
        """
        Refine a query to make it more specific and searchable.
        """
        prompt = f"""
        Refine this query to make it more specific and searchable. 
        If provided, consider the document context for relevant terminology.

        Original Query: "{query}"
        Document Context: {doc_context[:200]}

        Return only the refined query.
        """
        try:
            return self.llm.predict(prompt).strip()
        except Exception as e:
            logger.warning(f"Query refinement failed: {str(e)}")
            return query


### DocumentReranker Class ###
class DocumentReranker:
    """Re-ranks retrieved documents based on relevance using an LLM."""

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=api_key, model=model, temperature=0.0
        )

    def rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Re-rank documents with enhanced metadata and relevance scoring.
        """
        if not documents:
            return []

        try:
            prompt = f"""
            Rate these documents' relevance to the query (1-10):
            Query: {query}

            Documents:
            {self._format_documents(documents)}

            Return JSON: {{"scores": [n1, n2, ...], "reasoning": ["r1", "r2", ...]}}.
            """
            response = self.llm.predict(prompt)
            result = json.loads(response)

            for doc, score, reason in zip(documents, result["scores"], result["reasoning"]):
                doc.metadata.update(
                    {"relevance_score": score, "ranking_reason": reason, "query": query}
                )

            return sorted(documents, key=lambda d: d.metadata["relevance_score"], reverse=True)
        except Exception as e:
            logger.error(f"Re-ranking failed: {str(e)}")
            return documents

    @staticmethod
    def _format_documents(documents: List[Document]) -> str:
        return "\n\n".join(
            f"Doc {i + 1}:\n{doc.page_content[:300]}..." for i, doc in enumerate(documents)
        )


### VectorStoreManager Class ###
class VectorStoreManager:
    """Manages vector store creation and updates."""

    def __init__(self, source_dir: str, persist_dir: str, api_key: str):
        self.source_dir = source_dir
        self.persist_dir = persist_dir
        self.api_key = api_key
        self.vector_store = None

    def create_vector_store(self) -> Chroma:
        """
        Create or load a Chroma vector store.
        """
        embeddings = GoogleGenerativeAIEmbeddings(google_api_key=self.api_key, model="models/embedding-001")

        # Load or initialize the vector store
        if Path(self.persist_dir).exists() and len(list(Path(self.persist_dir).iterdir())) > 0:
            self.vector_store = Chroma(persist_directory=self.persist_dir, embedding_function=embeddings)
            logger.info("Loaded existing vector store.")
        else:
            self.vector_store = Chroma(persist_directory=self.persist_dir, embedding_function=embeddings)
            logger.info("Initialized new vector store.")

        # Update the vector store with new documents if necessary
        self.update_vector_store()
        return self.vector_store

    def update_vector_store(self):
        documents = load_existing_files(self.source_dir)
    # Filter out documents that are already in the store 
    # using some ID or hash stored in doc.metadata
        new_docs = []
        for doc in documents:
            if not self.already_in_store(doc):
                new_docs.append(doc)

        if new_docs:
            chunks = chunk_documents(new_docs)
        if chunks:
            self.vector_store.add_documents(chunks)
            logger.info(f"Added {len(chunks)} new chunks to the vector store.")
        else:
            logger.info("No new documents to add.")
        
        
    def already_in_store(self, doc: Document) -> bool:
        """
        Checks if a given Document is already present in the vector store.
        You must define a consistent way to identify a document.
        """
        # Example approach: Use a metadata field "file_name" or "doc_id" as a unique ID
        doc_id = doc.metadata.get("file_name")  # or some unique identifier



        # If your vector store supports filtering by metadata or doc_id, you might do:
        existing_docs = self.vector_store.similarity_search(doc_id, k=1)
        if not existing_docs:
            return False

        # Or you might store a hash/fingerprint somewhere and compare:
        # Return True if there's a match, else False.
        # The exact logic depends on your doc identification method.

        return True



### RAGChainProcessor Class ###
class RAGChainProcessor:
    """Manages the RAG pipeline for retrieving and answering user queries."""

    def __init__(self, vector_store: Chroma, api_key: str, llm_model: str = "gemini-1.5-flash"):
        self.vector_store = vector_store
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=api_key, model=llm_model, temperature=0.5, max_tokens=1000
        )
        self.prompt = PromptTemplate(template="""
            Answer the question based on the provided context.
            Context: {context}
            Question: {question}
            Answer:
        """)

    def process_query(self, query: str) -> str:
        """
        Retrieve documents and generate a response for the query.
        """
        retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        retrieved_docs = retriever.get_relevant_documents(query)

        if not retrieved_docs:
            return "I couldn't find any relevant information in the documents."

        context = "\n\n".join(f"Document {i + 1}:\n{doc.page_content}" for i, doc in enumerate(retrieved_docs))
        return self.llm.predict(self.prompt.format(context=context, question=query))


### QueryProcessor Class ###
class QueryProcessor:
    """Combines query validation, refinement, re-ranking, and the RAG pipeline."""

    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        rag_pipeline: RAGChainProcessor,
        query_refiner: QueryRefiner,
        reranker: DocumentReranker,
        llm: ChatGoogleGenerativeAI,
    ):
        self.validator = QueryValidator()
        self.vector_store_manager = vector_store_manager
        self.rag_pipeline =  rag_pipeline
        self.query_refiner = query_refiner
        self.reranker = reranker
        self.llm = llm  # Use LLM for generic queries

    def is_generic_query(self, query: str) -> bool:
        """
        Determines if a query is generic or document-related.
        :param query: The user query to classify.
        :return: True if the query is generic, False otherwise.
        """
        generic_patterns = [
            r"^(hi|hello|hey)[\s!]*$",
            r"^how are you",
            r"^what('s| is) up",
            r"^tell me a joke",
            r"^what is your name",
        ]
        return any(re.match(pattern, query.lower()) for pattern in generic_patterns)

    def process_query(self, query: str) -> str:
        """
        Validates, classifies, and processes the query through the appropriate pipeline.
        :param query: The user query to process.
        :return: The response or a validation error message.
        """
        # Step 1: Validate the query
        validation_result = self.validator.validate_query(query)
        if not validation_result["valid"]:
            return f"Query validation failed: {validation_result['reason']}"

        # Step 2: Check if the query is generic
        if self.is_generic_query(query):
            try:
                logger.info("Processing generic query.")
                return self.llm.predict(query)
            except Exception as e:
                logger.error(f"Error processing generic query: {str(e)}")
                return "Sorry, I couldn't process your request at this time."

        # Step 3: Refine the query using document context
        sample_docs = self.vector_store_manager.vector_store.similarity_search(query, k=2)
        document_context = " ".join(doc.page_content[:100] for doc in sample_docs)
        refined_query = self.query_refiner.refine_query(query, document_context)

        # Step 4: Retrieve and rerank documents
        retrieved_docs = self.vector_store_manager.vector_store.as_retriever().get_relevant_documents(refined_query)
        ranked_docs = self.reranker.rerank_documents(refined_query, retrieved_docs)

        # Step 5: Format top-ranked context and generate response
        context = "\n\n".join(f"[{i + 1}] {doc.page_content}" for i, doc in enumerate(ranked_docs[:3]))
        try:
            # return self.rag_pipeline.generate_response(context=context, question=refined_query) #edited
            return self.rag_pipeline.process_query(refined_query)#(context=context, question=refined_query)
        except Exception as e:
            logger.error(f"Error processing document-related query: {str(e)}")
            return "An error occurred while processing your query."

