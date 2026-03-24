from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

docs = [
    Document(
        page_content="""The Grand Canyon is one of the most visited natural wonders in the world. Photosynthesis is the process by which green plants convert sunlight into energy. Millions of tourists travel to see it every year. The rocks date back millions of years.""",
        metadata={"source": "Doc1"},
    ),
    Document(
        page_content="""In medieval Europe, castles were built primarily for defense. The chlorophyll in plant cells captures sunlight during photosynthesis. Knights wore armor made of metal. Siege weapons were often used to breach castle walls.""",
        metadata={"source": "Doc2"},
    ),
    Document(
        page_content="""Basketball was invented by Dr. James Naismith in the late 19th century. It was originally played with a soccer ball and peach baskets. NBA is now a global league.""",
        metadata={"source": "Doc3"},
    ),
    Document(
        page_content="""The history of cinema began in the late 1800s. Silent films were the earliest form. Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells. Modern filmmaking involves complex CGI and sound design.""",
        metadata={"source": "Doc4"},
    ),
]


embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
)
base_retriever = vector_store.as_retriever(search_kwargs={"k": 2})
llm = ChatOpenAI(model="gpt-4o-mini")
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=base_retriever
)
query = "What is photosynthesis?"
results = compression_retriever.invoke(query)
for i, doc in enumerate(results):
    print(f"-----Result {i+1}-----")
    print(f"Content:\n {doc.page_content}")


"""
Contextual Compression Retriever – Short Notes (.txt)
1. Problem (with real example)

In normal retrieval, the system returns complete documents or chunks, even if only a small part is relevant.

👉 Real Example:
User query:
"How to improve mental clarity?"

Retrieved document:

Walking improves heart health
Eating fruits improves longevity
Mindfulness improves mental clarity ✅
Drinking water boosts energy

❌ Problem:

Most of the content is irrelevant
LLM has to process unnecessary data
Leads to higher cost + lower accuracy
2. Solution – Contextual Compression Retrieval

Contextual Compression Retriever solves this by keeping only relevant parts of documents.

👉 How it works:

Step 1: User gives a query
Step 2: Retriever fetches relevant documents
Step 3: A compressor (LLM or filter) analyzes each document
Step 4: It removes irrelevant sentences/parts
Step 5: Only query-related content is kept
Step 6: Compressed results are returned

👉 Result:

Less noise
Better accuracy
Lower token usage
3. Example (after applying solution)

Query:
"How to improve mental clarity?"

Before Compression:

Full document with mixed information ❌

After Contextual Compression:

"Mindfulness and controlled breathing improve mental clarity" ✅

👉 Only the useful part is returned

4. Final Summary

Contextual Compression Retriever improves retrieval by filtering out unnecessary information and returning only the most relevant content, making responses more accurate, efficient, and cost-effective.
"""
