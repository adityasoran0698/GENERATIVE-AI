import os
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

load_dotenv()


embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("BASE_URL"),
)

user_query = input("👉 ")
vector_db = QdrantVectorStore.from_existing_collection(
    collection_name="RAG_SYSTEM", embedding=embedding_model, url="http://localhost:6333"
)

search_results = vector_db.similarity_search(user_query)
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("BASE_URL"),
)
context = [
    {
        "Page content": result.page_content,
        "Page number": result.metadata["page_label"],
        "Location": result.metadata["source"],
    }
    for result in search_results
]
SYSTEM_PROMPT = f"""
You are an helpful AI assistant which answers the user query based on the available context.

The available context is from pdf file having page number,content,file location. You have to answer about the available context on the basis of user query and navigate the user through page number to know more.

Available context:
{context}
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ],
)

print(response.choices[0].message.content)
