from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

doc1 = Document(
    page_content="Climate change is causing glaciers to melt rapidly in the Arctic region.",
    metadata={"source": "climate_1"},
)

doc2 = Document(
    page_content="Glaciers in the Arctic are melting at an alarming rate due to rising temperatures.",
    metadata={"source": "climate_2"},
)

doc3 = Document(
    page_content="Deforestation in the Amazon is accelerating global climate change.",
    metadata={"source": "climate_3"},
)

doc4 = Document(
    page_content="Climate change is increasing the frequency of wildfires in California.",
    metadata={"source": "climate_4"},
)

doc5 = Document(
    page_content="Rising sea levels due to climate change threaten coastal cities like Mumbai and New York.",
    metadata={"source": "climate_5"},
)

docs = [doc1, doc2, doc3, doc4, doc5]

vector_store = Chroma.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
    collection_name="Sample",
)
query = input("You: ")
retriever = vector_store.as_retriever(kwargs={"k": 3})
results = retriever.invoke(query)
for i, doc in enumerate(results):
    print(f"-----Result {i+1}-----")
    print(f"Content:\n {doc.page_content}")
