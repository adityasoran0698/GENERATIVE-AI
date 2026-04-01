from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

parser=StrOutputParser()
prompt1 = PromptTemplate(
    template="Write a detailed report on {topic}", input_variables=["topic"]
)
prompt2=PromptTemplate(
    template="Generate five pointer summary from the following text: \n {text}"
)
chain=prompt1 | model | parser | prompt2 | model | parser
result=chain.invoke({"topic":"cricket"})

print(result)
chain.get_graph().print_ascii()