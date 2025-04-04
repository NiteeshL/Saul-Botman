import requests
import time
import logging
import json
from haystack.nodes import BaseRetriever
from haystack.document_stores import FAISSDocumentStore
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("retriever")

# Hugging Face API configuration with fallback models
EMBEDDING_MODELS = [
    "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2",
    "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-mpnet-base-v2",
    "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
]

QA_MODELS = [
    "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2", 
    "https://api-inference.huggingface.co/models/distilbert-base-cased-distilled-squad"
]

# Important: Make sure the token format is correct - it should begin with "hf_"
# DO NOT include "Bearer " prefix in the token string itself
HEADERS = {"Authorization": "Bearer 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"}

# Load FAISS document store
document_store = FAISSDocumentStore.load(index_path="faiss_index", config_path="faiss_index.json")

class HuggingFaceRetriever(BaseRetriever):
    def __init__(self, document_store):
        super().__init__()
        self.document_store = document_store
        self.current_embedding_model_index = 0

    def get_embedding(self, query: str, max_retries: int = 3) -> List[float]:
        """Try to get embeddings from multiple models until one succeeds"""
        
        for model_index in range(len(EMBEDDING_MODELS)):
            self.current_embedding_model_index = model_index
            model_url = EMBEDDING_MODELS[model_index]
            model_name = model_url.split("/")[-1]
            
            logger.info(f"Trying embedding model {model_index+1}/{len(EMBEDDING_MODELS)}: {model_name}")
            
            for attempt in range(max_retries):
                try:
                    logger.info(f"Attempt {attempt+1}: Sending request to embedding model...")
                    
                    # For sentence-transformers, use this format
                    response = requests.post(
                        model_url, 
                        headers=HEADERS, 
                        json={"inputs": query, "options": {"wait_for_model": True}}
                    )
                    
                    logger.info(f"API Response Status: {response.status_code}")
                    
                    if response.status_code == 200:
                        logger.info("Successfully got response from embedding model")
                        result = response.json()
                        
                        # Extract embedding from response
                        if isinstance(result, list):
                            if len(result) > 0:
                                return result[0]
                        elif isinstance(result, dict):
                            if "embeddings" in result and len(result["embeddings"]) > 0:
                                return result["embeddings"][0]
                            elif "embedding" in result:
                                return result["embedding"]
                        
                        logger.error(f"Couldn't extract embedding from response: {str(result)[:100]}...")
                        break
                    
                    elif response.status_code == 401 or response.status_code == 403:
                        logger.error(f"Authentication error: {response.text}")
                        # No point retrying with same credentials
                        break
                    
                    elif response.status_code == 503:
                        # Model is loading
                        logger.info("Model is loading, waiting before retry...")
                        time.sleep(10)  # Wait before retrying
                        continue
                    
                    else:
                        logger.warning(f"Error response: {response.text}")
                        time.sleep(5)
                        continue
                        
                except Exception as e:
                    logger.error(f"Exception during embedding request: {str(e)}")
                    time.sleep(5)
            
        # If we get here, all models failed
        raise Exception("All embedding models failed")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        try:
            # Get embeddings from one of our models
            query_embedding = self.get_embedding(query)
            
            # Log successful embedding generation
            logger.info(f"Successfully generated embedding with {len(query_embedding)} dimensions")
            
            # Use the embedding to retrieve documents
            return self.document_store.query_by_embedding(query_embedding, top_k=top_k)
                
        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}")
            raise e

    def retrieve_batch(self, queries: List[str], top_k: int = 5) -> List[List[Dict[str, Any]]]:
        return [self.retrieve(query, top_k) for query in queries]

def fetch_relevant_documents(query: str) -> Dict[str, Any]:
    try:
        logger.info(f"Processing query: {query}")
        
        # Use retriever to get relevant documents
        retriever = HuggingFaceRetriever(document_store)
        docs = retriever.retrieve(query, top_k=5)
        
        # Log successful document retrieval
        logger.info(f"Retrieved {len(docs)} documents")
        
        # Use QA model API to get answers - try models in order until one works
        context = " ".join([doc.content for doc in docs])
        qa_result = None
        
        for model_url in QA_MODELS:
            model_name = model_url.split("/")[-1]
            logger.info(f"Trying QA model: {model_name}")
            
            try:
                qa_payload = {
                    "inputs": {
                        "question": query,
                        "context": context
                    },
                    "options": {
                        "wait_for_model": True
                    }
                }
                
                qa_response = requests.post(model_url, headers=HEADERS, json=qa_payload, timeout=30)
                
                if qa_response.status_code == 200:
                    qa_result = qa_response.json()
                    logger.info("Successfully got answer from QA model")
                    break
                else:
                    logger.warning(f"QA model error: Status {qa_response.status_code}, Response: {qa_response.text}")
                    continue
                    
            except Exception as e:
                logger.error(f"Error with QA model {model_name}: {str(e)}")
                continue
        
        if qa_result is None:
            logger.warning("All QA models failed, returning just the documents")
            
        return {
            "query": query,
            "answers": [qa_result] if qa_result else [],
            "documents": docs
        }
    except Exception as e:
        logger.error(f"Error in fetch_relevant_documents: {str(e)}")
        return {
            "query": query,
            "error": str(e),
            "documents": []
        }

# Simple test function for checking token validity
def test_token():
    logger.info("Testing authentication with Hugging Face API...")
    response = requests.post(
        "https://huggingface.co/api/whoami", 
        headers=HEADERS
    )
    
    if response.status_code == 200:
        user_info = response.json()
        logger.info(f"Authentication successful. User: {user_info.get('name', 'Unknown')}")
        return True
    else:
        logger.error(f"Authentication failed. Status code: {response.status_code}")
        logger.error(f"Response: {response.text}")
        return False