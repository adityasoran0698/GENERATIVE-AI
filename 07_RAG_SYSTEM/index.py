import os
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from pydantic_core import Url

load_dotenv()

# Getting the PDF file path
pdf_path = Path(__file__).parent / "07_RAG" / "notes.pdf"

# Loading the PDF file
loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()

# Split the PDF into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
chunks = text_splitter.split_documents(documents=docs)

# Vector Embeddings

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

vector_db = QdrantVectorStore.from_documents(
    documents=chunks,
    url="http://localhost:6333",
    embedding=embedding_model,
    collection_name="RAG_SYSTEM",
)
