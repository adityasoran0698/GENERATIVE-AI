from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from time import time

load_dotenv()

loader = DirectoryLoader(path="PDFS", glob="*.pdf", loader_cls=PyMuPDFLoader)

# docs = loader.load()
docs = loader.lazy_load()
for doc in docs:
    print(doc.metadata)
