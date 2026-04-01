from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

loader=TextLoader(file_path="cricket.txt",encoding="utf-8")
docs=loader.load()
model=ChatOpenAI(model="gpt-4o-mini")
parser=StrOutputParser()
prompt1=PromptTemplate(
    template="Summarise the poem.\nPoem: {poem}",
    input_variables=['poem']
)
chain=prompt1 | model | parser
result=chain.invoke({"poem":docs[0].page_content})
print(result)