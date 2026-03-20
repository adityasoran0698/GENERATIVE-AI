from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

# 1st Prompt

template1 = PromptTemplate(
    template="Write a detailed report on {topic}", input_variables="topic"
)
topic = input("Enter topic: ")
prompt1 = template1.invoke({"topic": topic})

result1 = model.invoke(prompt1)

# 2nd Prompt
template2 = PromptTemplate(
    template="Summarise the give text in 5 points \n Text: {text}",
    input_variables="text",
)
prompt2 = template2.invoke({"text": result1.content})

result2 = model.invoke(prompt2)
print(result2.content)
