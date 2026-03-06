# Import OpenAI client to communicate with Gemini API using OpenAI-style syntax
from openai import OpenAI

# Import dotenv to load environment variables from .env file
from dotenv import load_dotenv

# Import os module to access environment variables
import os

# Load variables from .env file into environment
load_dotenv()


# Create a client object with API key and Gemini base URL
client = OpenAI(
    api_key=os.getenv(
        "OPENAI_API_KEY"
    ),  # Fetch API key securely from environment variable
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",  # Gemini OpenAI-compatible endpoint
)


# Send a request to the model using chat completions API
response = client.chat.completions.create(
    model="gemini-2.5-flash",  # Specify which Gemini model to use
    messages=[
        # System message: defines the behavior/personality of the AI (Prompt Engineering concept)
        {
            "role": "system",
            "content": "You are a great mathematician that answers only those questions which are related to maths only. Otherwise you will say sorry",
        },
        # User message: the actual question asked by the user
        {"role": "user", "content": "what is photosynthesis"},
    ],
)

# Print only the model's final reply from the response object
print(response.choices[0].message.content)
