from haystack.pipelines import ExtractiveQAPipeline
from haystack.nodes import DensePassageRetriever, FARMReader
from haystack.document_stores import FAISSDocumentStore

# Load FAISS document store
document_store = FAISSDocumentStore.load(index_path="faiss_index")

# Initialize retriever and reader
retriever = DensePassageRetriever(
    document_store=document_store,
    query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
    passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
    use_gpu=True
)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")

# Create QA pipeline
pipeline = ExtractiveQAPipeline(reader, retriever)

def fetch_relevant_documents(query):
    return pipeline.run(query=query, params={"Retriever": {"top_k": 5}, "Reader": {"top_k": 3}})
