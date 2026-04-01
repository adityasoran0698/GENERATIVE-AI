from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

parser = StrOutputParser()

model = ChatOpenAI(model="gpt-4o-mini")

# 1st Prompt
topic = input("Enter topic: ")

template1 = PromptTemplate(
    template="Write a detailed report on {topic}", input_variables="topic"
)
# 2nd Prompt
template2 = PromptTemplate(
    template="Summarise the give text in 5 points \n Text: {text}",
    input_variables="text",
)

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic": topic})
print(result)
