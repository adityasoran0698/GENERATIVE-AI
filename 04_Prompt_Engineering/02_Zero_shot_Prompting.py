from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("BASE_URL")
)

# Zero shot Prompting
System_prompt = """
Your name is Alice. You answers questions only that are related to coding. If its not, you will simply avoid answering it by telling user in  respectfully,guilty,and friendly way.
"""
while True:
    user_input = input("Enter")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": System_prompt},
            {"role": "user", "content": user_input},
        ],
    )
    print(response.choices[0].message.content)
