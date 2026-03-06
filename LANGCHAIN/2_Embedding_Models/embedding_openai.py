from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()

embedding=OpenAIEmbeddings(model='text-embedding-3-large', api_key=os.getenv("OPENAI_API_KEY"),base_url=os.getenv("BASE_URL"),dimensions=32,)

docs=[
    "Delhi is capital of india",
    "Mumbai is financial capital of india",
    "Bangalore is IT capital of india",
    "Chennai is cultural capital of india",
]
result=embedding.embed_documents(docs)
print(result)