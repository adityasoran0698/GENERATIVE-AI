from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
load_dotenv()


documents = [
    "Virat Kohli is an Indian cricketer and former captain of the Indian national team. He is widely regarded as one of the greatest batsmen in the history of cricket.",
    "Sachin Tendulkar is a former Indian cricketer and one of the greatest batsmen in the history of cricket. He is often referred to as the 'God of Cricket' and has numerous records to his name.",
    "M.S. Dhoni is a former Indian cricketer and captain of the Indian national team. He is known for his calm demeanor and exceptional leadership skills, leading India to several victories in international cricket.",
    "Rohit Sharma is an Indian cricketer and the current captain of the Indian national team. He is known for his aggressive batting style and has set several records in limited-overs cricket.",
    "Jasprit Bumrah is an Indian cricketer and one of the best fast bowlers in the world. He is known for his unique bowling action and ability to bowl yorkers consistently.",
]
embedding = OpenAIEmbeddings(
    model="text-embedding-3-large",
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    dimensions=300,
)
user_query = input("👉 ")


doc_embedding = embedding.embed_documents(documents)
user_embedding = embedding.embed_query(user_query)

scores=cosine_similarity([user_embedding], doc_embedding)[0]
similarity=sorted(list(enumerate(scores)),key=lambda x:x[1])
index,score=similarity[-1]
print(documents[index])