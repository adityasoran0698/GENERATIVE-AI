from langchain_community.document_loaders import CSVLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from time import time

load_dotenv()

start = time()
loader = CSVLoader("sample.csv")

docs = list(loader.lazy_load())
# Takes less time becuase it does not loads all the documents at once. It load one document at once, and then remove that document,then loads the next one. Suppose there are 500 document so it loads one document at once then remove that then loads the next one and so on...

#  docs=loader.lazy_load() ->Takes more time becuase it loads all the documents at once. Suppose there are 500 document so it loads all 500 at once.
print(docs)
print(docs[233].page_content)
print(docs[233].metadata)
end = time()
print(f"{end-start:.2f}s")
