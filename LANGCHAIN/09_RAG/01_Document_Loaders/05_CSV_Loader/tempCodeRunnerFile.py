from langchain_community.document_loaders import CSVLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

loader=CSVLoader("sample.csv")
docs=loader.lazy_load()
print(docs)
print(docs[233].page_content)
print(docs[233].metadata)