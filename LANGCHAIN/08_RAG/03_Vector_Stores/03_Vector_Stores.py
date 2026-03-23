from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

doc1 = Document(
    page_content="Virat Kohli is one of the best batsmen in the world. He has scored many centuries in international cricket and is known for his aggressive playing style.",
    metadata={"player": "Virat Kohli", "role": "Batsman", "id": 1},
)

doc2 = Document(
    page_content="Rohit Sharma is the captain of the Indian cricket team in limited overs. He is famous for hitting big sixes and has multiple double centuries in ODI cricket.",
    metadata={"player": "Rohit Sharma", "role": "Batsman", "id": 2},
)

doc3 = Document(
    page_content="MS Dhoni is one of the most successful captains of India in cricket. He is known for his calm nature and finishing matches under pressure.",
    metadata={"player": "MS Dhoni", "role": "Wicketkeeper and Batsman", "id": 3},
)

doc4 = Document(
    page_content="Jasprit Bumrah is a fast bowler known for his unique bowling action and deadly yorkers. He plays a key role in India's bowling attack.",
    metadata={"player": "Jasprit Bumrah", "role": "Bowler", "id": 4},
)

doc5 = Document(
    page_content="Hardik Pandya is an all-rounder who contributes with both bat and ball. He is known for his power hitting and fast bowling.",
    metadata={"player": "Hardik Pandya", "role": "All-rounder", "id": 5},
)

docs = [doc1, doc2, doc3, doc4, doc5]


vector_store = Chroma(
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"),
    persist_directory="chroma_db",
    collection_name="sample",
)
vector_store.add_documents(docs)
# result=vector_store.get(include=['embeddings','documents','metadatas'])
query = input("You: ")
result1 = vector_store.similarity_search(query=query,k=5)

model = ChatOpenAI(model="gpt-4o-mini")

prompt1 = PromptTemplate(
    template="""You are an assistant. Answer ONLY from the given context.
If the answer is not in the context, say "I don't know".

Query: {query}

Context:
{document}
"""
)
parser = StrOutputParser()
context = "\n\n".join([doc.page_content for doc in result1])

chain = prompt1 | model | parser
result2 = chain.invoke({"query": query, "document": context})

print(result2)
