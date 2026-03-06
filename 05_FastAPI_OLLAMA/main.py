from fastapi import FastAPI,Body
from ollama import Client

app = FastAPI()
client = Client(host="http://localhost:11434")

@app.get("/")
def read():
    return {"message": "Hello world"}

@app.get("/home")
def func():
    return{"message":"Homepage"}

@app.post("/chat")
def Post(message:str=Body(...,description="The message")):
    response=client.chat(model="gemma:2b",messages=[
        {"role":"user","content":message}
    ])
    return{"response":response.message.content}
