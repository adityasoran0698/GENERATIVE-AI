from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import pipeline

pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
llm = HuggingFacePipeline(pipeline=pipe)

model = ChatHuggingFace(llm=llm)

result = model.invoke("what is Capital of india?")
print(result.content)
