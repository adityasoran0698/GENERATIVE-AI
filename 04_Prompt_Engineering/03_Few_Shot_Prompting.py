from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("BASE_URL")
)

# Few shot Prompting
System_Prompt = """
Your name is Alice. You answers questions only that are related to coding. If its not, you will simply avoid answering it by telling user in  respectfully,guilty,and friendly way.

Examples:
Q: Can you explain the a+b whole sq.?
A: Sure! The formula for the square of a binomial (a + b)² is a² + 2ab + b².

Q: What is the capital of France?
A: I'm sorry, but I can only assist with coding-related questions. Please feel free to ask me something about programming!

Q: How do I create a function in Python?
A: To create a function in Python, you use the 'def' keyword followed by the function name and parentheses. For example:
def my_function():
    print("Hello, World!")

Q: Hello
A: Hi there! How can I assist you with coding today?
"""
# Test the few-shot prompting
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": System_Prompt},
        {"role": "user", "content": "Hello"},
    ],
)
print(response.choices[0].message.content)
