import os
import requests
from haystack.document_stores import FAISSDocumentStore
from haystack.utils import clean_wiki_text, convert_files_to_docs

# Initialize FAISS document store
document_store = FAISSDocumentStore(embedding_dim=768, faiss_index_factory_str="Flat")

# Hugging Face API configuration for embeddings
EMBEDDING_API_URL = "https://api-inference.huggingface.co/models/facebook/dpr-ctx_encoder-single-nq-base"
HEADERS = {"Authorization": "Bearer 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"}

def ingest_documents(data_dir):
    # Convert legal documents to Haystack format
    docs = convert_files_to_docs(dir_path=data_dir, clean_func=clean_wiki_text, split_paragraphs=True)

    def generate_embeddings(docs):
        for doc in docs:
            payload = {"inputs": doc['content']}
            response = requests.post(EMBEDDING_API_URL, headers=HEADERS, json=payload)
            response.raise_for_status()
            doc['embedding'] = response.json()['embedding']
        return docs

    docs = generate_embeddings(docs)
    document_store.write_documents(docs)

    # Save FAISS index and configuration
    document_store.save(index_path="faiss_index", config_path="faiss_index.json")

if __name__ == "__main__":
    data_dir = "./legal_documents"  # Path to legal documents
    ingest_documents(data_dir)
