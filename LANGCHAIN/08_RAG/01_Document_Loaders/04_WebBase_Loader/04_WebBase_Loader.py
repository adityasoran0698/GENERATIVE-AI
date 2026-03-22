from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


load_dotenv()
url = "https://webscraper.io/test-sites"
loader = WebBaseLoader(url)
model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()
prompt = PromptTemplate(
    template="I am providing you a content of a website. Analayse the content for answering the questions. \n site_content->{data}\n Question->{question}",
    input_variables=["data", "question"],
)
# docs = loader.load()
docs = loader.load()
chain = prompt | model | parser
result = chain.invoke(
    {"data": docs[0].page_content, "question": "What the site is about?"}
)

print(result)