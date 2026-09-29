import os
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# LangChain Imports (Updated & Error-Free imports)
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Environment Variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "dambulla_ds_circulars")

# Initialize FastAPI App
app = FastAPI(
    title="Dambulla Divisional Secretariat AI Assistant API",
    description="Multilingual RAG API for DS Office Services",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Multilingual Embedding Model
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 2. Qdrant Vector DB Connection
try:
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
except Exception as e:
    print(f"Warning: Qdrant client initialization error: {e}")
    retriever = None

# 3. Google Gemini Setup
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.2,
    max_tokens=1000
)

# 4. System Prompt
system_prompt = (
    "ඔබ දඹුල්ල ප්‍රාදේශීය ලේකම් කාර්යාලයේ නිල AI සහායකයා වේ.\n"
    "You are the official AI assistant for the Dambulla Divisional Secretariat.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Build Chain
document_chain = create_stuff_documents_chain(llm, prompt)
if retriever:
    rag_chain = create_retrieval_chain(retriever, document_chain)
else:
    rag_chain = None

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

@app.get("/")
def read_root():
    return {"status": "online", "office": "Dambulla Divisional Secretariat"}

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    if not rag_chain:
        raise HTTPException(status_code=500, detail="Database/Retrieval chain is not initialized.")

    try:
        response = rag_chain.invoke({"input": request.question})
        
        sources = []
        for doc in response.get("context", []):
            circular_no = doc.metadata.get("circular_no") or doc.metadata.get("source", "Official Guideline")
            if circular_no not in sources:
                sources.append(str(circular_no))

        return QueryResponse(
            answer=response["answer"],
            sources=sources
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Engine Error: {str(e)}")
