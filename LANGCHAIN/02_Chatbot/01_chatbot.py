from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model="gpt-4o-mini"
)

while True:
    user_query=input("You: ")
    if(user_query=="exit"):
        break
    response=model.invoke(user_query)
    print(f"AI: {response.content}")