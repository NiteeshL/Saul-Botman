import os
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import DensePassageRetriever
from haystack.utils import clean_wiki_text, convert_files_to_docs, fetch_archive_from_http

# Initialize FAISS document store
document_store = FAISSDocumentStore(embedding_dim=768, faiss_index_factory_str="Flat")

def ingest_documents(data_dir):
    # Convert legal documents to Haystack format
    docs = convert_files_to_docs(dir_path=data_dir, clean_func=clean_wiki_text, split_paragraphs=True)
    document_store.write_documents(docs)

    # Initialize retriever
    retriever = DensePassageRetriever(
        document_store=document_store,
        query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
        passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
        use_gpu=True
    )
    document_store.update_embeddings(retriever)

if __name__ == "__main__":
    data_dir = "./legal_documents"  # Path to legal documents
    ingest_documents(data_dir)
