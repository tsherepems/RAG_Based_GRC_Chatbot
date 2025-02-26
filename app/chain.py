
"""
app/chain.py
Manages the vector store creation and the simple RAG pipeline.
"""
import logging
import os
import re
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from app import  load_existing_files, chunk_documents, load_file,save_processed_cache, load_processed_cache,get_file_hash
from config import api_key
from langchain.schema import Document
from googleapiclient import discovery
from google.auth.credentials import AnonymousCredentials
from config import Perspective_api  

# Configure logging (using INFO level)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryValidator:
    """
    Validates a query using the Perspective API to detect offensive or profane content.
    """
    def validate_query(self, query: str) -> dict:
        # Basic checks for empty string, length, and invalid characters.
        if not query.strip():
            return {"valid": False, "reason": "Query is empty", "is_generic": False}
        if len(query) > 500:
            return {"valid": False, "reason": "Query is too long", "is_generic": False}
        if re.search(r"[<>$;]", query):
            return {"valid": False, "reason": "Query contains invalid characters", "is_generic": False}
        
        try:
            # Initialize the Perspective API client using the official discovery library.
            client = discovery.build(
                "commentanalyzer",
                "v1alpha1",
                developerKey=Perspective_api,
                credentials=AnonymousCredentials(),
                discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
                static_discovery=False,
            )
            
            # Build the request body
            analyze_request = {
                "comment": {"text": query},
                "languages": ["en"],
                "requestedAttributes": {"TOXICITY": {}, "PROFANITY": {}},
                "doNotStore": True
            }
            
            # Execute the API request
            response = client.comments().analyze(body=analyze_request).execute()
            
            # Extract toxicity and profanity scores
            toxicity = response.get("attributeScores", {}) \
                               .get("TOXICITY", {}) \
                               .get("summaryScore", {}) \
                               .get("value", 0)
            profanity = response.get("attributeScores", {}) \
                                .get("PROFANITY", {}) \
                                .get("summaryScore", {}) \
                                .get("value", 0)
            
            # Check scores against the threshold (0.7)
            if toxicity >= 0.7 or profanity >= 0.7:
                return {"valid": False, "reason": "Query contains offensive or inappropriate content.", "is_generic": False}
            
        except Exception as e:
            logger.warning("Error calling Perspective API: %s", e)
        
        return {"valid": True, "reason": "", "is_generic": False}



class VectorStoreManager:
    """
    Handles creating or loading the Chroma vector store and updating it with new documents.
    """

    def __init__(self, source_dir: str, persist_dir: str):
        self.source_dir = source_dir
        self.persist_dir = persist_dir
        self.api_key = api_key
        self.vector_store = None

    def create_or_load_vector_store(self) -> Chroma:
        """
        Creates or loads an existing Chroma store, then updates it with any new documents.
        """
        embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=self.api_key, model="models/embedding-001"
        )

        # If the persist directory has data, load from it; otherwise create a new store
        if Path(self.persist_dir).exists() and any(Path(self.persist_dir).iterdir()):
            logger.info("Loading existing Chroma vector store...")
            self.vector_store = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=embeddings,
            )
        else:
            logger.info("Initializing new Chroma vector store...")
            self.vector_store = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=embeddings,
            )

        # Update with any new documents from source_dir
        self.update_vector_store()
        return self.vector_store
    
    def update_vector_store(self):
        """ Loads files from the source directory, checks if they are new or changed by comparing hashes,
        and adds only new document chunks to the vector store. """
        
        processed_cache = load_processed_cache()  # Load our persistent cache
        new_documents = []
        
        # Iterate through all files in the source directory
        for file_path in Path(self.source_dir).rglob("*"):
            if file_path.is_file():
                file_str = str(file_path)
                current_hash = get_file_hash(file_str)
                
                # If the file has been processed before and the hash is the same, skip it.
                if file_str in processed_cache and processed_cache[file_str] == current_hash:
                    logging.info(f"Skipping already processed file: {file_str}")
                    continue
                
                # Otherwise, process the file.
                try:
                    text = load_file(file_str)
                    if text.strip():
                        new_documents.append(Document(page_content=text, metadata={"source": file_str}))
                        # Update the cache with the new hash.
                        processed_cache[file_str] = current_hash
                except ValueError as ve:
                    logging.warning(f"Skipping file {file_str}: {ve}")
        
        # Save the updated cache
        save_processed_cache(processed_cache)
        
        # If there are new documents, chunk and add them to the vector store.
        if new_documents:
            chunks = chunk_documents(new_documents)
            if chunks:
                self.vector_store.add_documents(chunks)
                logging.info(f"Added {len(chunks)} new document chunks to the vector store.")




class RAGChain:
    """
    A simple RAG chain that retrieves relevant chunks and uses LLM to generate answers.
    """

    def __init__(self, vector_store: Chroma):
        self.vector_store = vector_store
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model="gemini-1.5-flash",
            temperature=0.5,
            max_tokens=1000
        )
        self.prompt_template = PromptTemplate(template="""
            You are a GRC (Governance, Risk, Compliance) expert tutor with years of experience. 
            Answer the question below using only the provided context. 
            If you don't find relevant info, say you are not sure.

            Context:
            {context}

            Question:
            {question}

            Answer:
        """, input_variables=["context", "question"])

    def run(self, query: str) -> str:
        retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        docs = retriever.invoke(query)

        if not docs:
            return "I'm not sure. I don't have relevant information in my documents."

        context_str = "\n\n".join([doc.page_content for doc in docs])
        prompt = self.prompt_template.format(context=context_str, question=query)
    
        response = self.llm.invoke(prompt)
    
        # Instead of calling strip() directly on response, extract text content if available.
        if hasattr(response, "content"):
            answer = response.content
        else:
            answer = str(response)
    
        return answer.strip()
    
class QueryRefiner:
    def refine_query(self, query: str, doc_context: str = "") -> str:
        prompt = f"""
        Refine the following query to make it more specific and better suited for document retrieval.
        Original Query: "{query}"
        Document Context: {doc_context[:200]}
        Return only the refined query.
        """
        try:
            refined = self.llm.invoke(prompt)
            if hasattr(refined, "content"):
                return refined.content.strip()
            return str(refined).strip()
        except Exception as e:
            logger.warning("Query refinement failed: %s", e)
            return query
        

class QueryProcessor: #edited with generic query feature

    def __init__(self, rag_chain: RAGChain):
        self.rag_chain = rag_chain
        self.validator = QueryValidator()
        self.query_refiner = QueryRefiner()



    def is_generic_query(self, query: str) -> bool:
            
        """
        Uses an LLM prompt to classify the query as generic or document-specific.
        """
        prompt = f"""
        Determine if the following query is generic (e.g., greetings, small talk, or simple conversation) 
        or document-specific (requiring contextual document retrieval). 
        Respond with only one word: "generic" or "document-specifi      
        Query: "{query}"
        """
        try:
            response = self.rag_chain.llm.invoke(prompt)
            if hasattr(response, "content"):
                classification = response.content.strip().lower()
            else:
                classification = str(response).strip().lower()
            return classification == "generic"
        except Exception as e:
            logger.warning("Generic query classification failed: %s", e)
            # Fallback to regex if LLM call fails.
            generic_patterns = [
                r"^(hi|hello|hey)[\s!]*$",
                r"^how are you",
                r"^what('s| is) up",
                r"^tell me a joke",
                r"^what is your name"
            ]
            return any(re.match(pattern, query.lower()) for pattern in generic_patterns)

    def process_query(self, query: str) -> str:
        """
        Processes the query in the following steps:
        1. Profanity check via the Perspective API.
        2. Generic query separation: if generic, directly call the LLM.
        3. For document-related queries, refine the query using document context.
        4. Retrieve and answer using the RAG chain.
        """
        if not query.strip():
            return "Please provide a valid query."

        # Step 1: Validate the query
        validation = self.validator.validate_query(query)
        if not validation["valid"]:
            return f"Query blocked: {validation['reason']}"

        # Step 2: Generic query check
        if self.is_generic_query(query):
            try:
                logger.info("Processing generic query directly via LLM.")
                response = self.rag_chain.llm.invoke(query)
                if hasattr(response, "content"):
                    return response.content.strip()
                return str(response).strip()
            except Exception as e:
                logger.error("Error processing generic query: %s", e)
                return "Sorry, I couldn't process your request at this time."

        # Step 3: Document-specific query refinement
        # Retrieve a small sample of document context for refinement.
        retriever = self.rag_chain.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
        sample_docs = retriever.invoke(query)
        doc_context = " ".join([doc.page_content[:100] for doc in sample_docs]) if sample_docs else ""
        refined_query = self.query_refiner.refine_query(query, doc_context)
        logger.info("Refined Query: %s", refined_query)

        # Step 4: Use the refined query for document retrieval and answer generation.
        return self.rag_chain.run(refined_query)
