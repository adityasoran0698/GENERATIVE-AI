from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

loader = PyPDFLoader("JAVA_OOPs.pdf")
docs = loader.load()
print(docs)
# model = ChatOpenAI(model="gpt-4o-mini")
# parser = StrOutputParser()

