from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
import requests
from dotenv import load_dotenv

load_dotenv()


@tool
def multiply(a: int, b: int) -> int:
    "Given two number a and b. This tool return their product"
    return a * b


# Tool Binding
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools([multiply])
# print(llm_with_tools)


# Tool Calling 
result=llm_with_tools.invoke("Multiply 8 by 67")
# print(result)
# print(result.tool_calls[0])

