from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_classic.retrievers import MultiQueryRetriever
from dotenv import load_dotenv

load_dotenv()

all_docs = [
    Document(
        page_content="Regular walking boosts heart health and can reduce symptoms of depression.",
        metadata={"source": "H1"},
    ),
    Document(
        page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.",
        metadata={"source": "H2"},
    ),
    Document(
        page_content="Deep sleep is crucial for cellular repair and emotional regulation.",
        metadata={"source": "H3"},
    ),
    Document(
        page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.",
        metadata={"source": "H4"},
    ),
    Document(
        page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.",
        metadata={"source": "H5"},
    ),
    Document(
        page_content="The solar energy system in modern homes helps balance electricity demand.",
        metadata={"source": "I1"},
    ),
    Document(
        page_content="Python balances readability with power, making it a popular system design language.",
        metadata={"source": "I2"},
    ),
    Document(
        page_content="Photosynthesis enables plants to produce energy by converting sunlight.",
        metadata={"source": "I3"},
    ),
    Document(
        page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.",
        metadata={"source": "I4"},
    ),
    Document(
        page_content="Black holes bend spacetime and store immense gravitational energy.",
        metadata={"source": "I5"},
    ),
]

vector_store = Chroma.from_documents(
    documents=all_docs,
    embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
)
query = "How to improve energy levels and maintain balance?"

similarity_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
)

similarity_result = similarity_retriever.invoke(query)
multiquery_result = multiquery_retriever.invoke(query)
multiquery_result = multiquery_result[:5]
for i, doc in enumerate(similarity_result):
    print(f"-----Result {i+1}-----")
    print(f"Content:\n {doc.page_content}")
print("\n\n")


for i, doc in enumerate(multiquery_result):
    print(f"-----Result {i+1}-----")
    print(f"Content:\n {doc.page_content}")


"""
MultiQuery Retriever – Short Notes
1. Problem (with real example)

In normal retrieval, only one query is used to search documents.
If the query is ambiguous or incomplete, the retriever may return incorrect or limited results.

👉 Real Example:
User query:
"How to improve energy?"

Problem:

"Energy" can mean:
Human energy (health) ✅
Electrical energy ❌
Scientific energy ❌

Normal Retriever Output:

Drinking water improves energy ✅
Solar energy system ❌
Black holes store energy ❌

❌ Problem:

Mixed and irrelevant results
Some important health-related documents may be missed

2. Solution – MultiQuery Retrieval

MultiQuery Retriever solves this by generating multiple versions of the same query using an LLM.

👉 How it works:

Step 1: User gives a query
Step 2: LLM creates multiple related queries
Step 3: Each query searches the vector database
Step 4: Results from all queries are combined
Step 5: Duplicate documents are removed
Step 6: Final diverse and relevant results are returned

👉 Result:

Better understanding of user intent
Covers multiple meanings of query
Improves recall (finds more relevant docs)

3. Example (after applying solution)

Query:
"How to improve energy?"

Generated Queries:

"How to improve physical energy levels?"
"Ways to stay active and healthy"
"Tips for better metabolism and stamina"

Final Output:

Drinking water improves energy ✅
Exercise improves health ✅
Healthy diet improves energy ✅

👉 Irrelevant results (solar, physics) are reduced

4. Final Summary

MultiQuery Retriever improves retrieval by expanding a single query into multiple queries, searching from different perspectives, and combining results to provide more accurate and comprehensive answers.
"""
