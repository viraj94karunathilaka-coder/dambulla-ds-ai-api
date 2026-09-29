import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant

# Load PDFs from a local folder named 'circulars'
loader = DirectoryLoader('./circulars', glob="./*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

# Split text into optimal chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

# Create Multilingual Embeddings
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

# Upload to Qdrant Cloud
QDRANT_URL = os.getenv("QDRANT_URL", "https://your-cluster-url.qdrant.tech:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "YOUR_QDRANT_API_KEY")

vector_store = Qdrant.from_documents(
    documents=chunks,
    embedding=embeddings,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    collection_name="dambulla_ds_circulars",
    force_recreate=True
)

print("Ingestion Complete! All circulars indexed to Qdrant Cloud.")