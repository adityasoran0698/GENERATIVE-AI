from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
message_history = []

model = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model="gpt-4o-mini",
)

while True:
    user_query = input("You: ")
    message_history.append(user_query)
    if user_query == "exit":
        break
    response = model.invoke(message_history)
    message_history.append(response.content)
    print(f"AI: {response.content}")

print(message_history)
