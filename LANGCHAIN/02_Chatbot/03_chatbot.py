from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini")


messages = [
    SystemMessage("You are a Helpful assistant. Call me Aditya."),
]
while True:

    user_query = input("You: ")
    if user_query == "exit":
        break
    messages.append(HumanMessage(user_query))
    response = model.invoke(messages)
    messages.append(AIMessage(response.content))
    print(response.content)

print(messages)
