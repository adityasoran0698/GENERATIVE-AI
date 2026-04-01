from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool, InjectedToolArg
from typing import Annotated
import requests
import json
from dotenv import load_dotenv

load_dotenv()


# Tool Creating
@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    "This Function fetches the conversion factor between a base currency and target currency"
    url = f"https://v6.exchangerate-api.com/v6/4e8bf3dfe1f6b8afc9f5f96e/pair/{base_currency}/{target_currency}"
    response = requests.get(url)
    return response.json()


@tool
def convert(
    base_currency_value: int, conversion_rate: Annotated[float, InjectedToolArg]
) -> float:
    "This Function converts the base currency value using conversion rate"
    return base_currency_value * conversion_rate


# Tool Binding
llm = ChatOpenAI()
llm_with_tools = llm.bind_tools([get_conversion_factor, convert])

# Tool Calling
messages = []
query = "What is the conversion factor between CNY currency and INR Currency and base on this convert 10 China currency  in to  India Currency "
messages.append(HumanMessage(query))
tool_result = llm_with_tools.invoke(messages)
messages.append(tool_result)

# Tool Execution
for tool_call in tool_result.tool_calls:
    # 1st tool call
    if tool_call["name"] == "get_conversion_factor":
        tool_message1 = get_conversion_factor.invoke(tool_call)
        # Fetching Conversion Rate
        conversion_rate = json.loads(tool_message1.content)["conversion_rate"]
        messages.append(tool_message1)

    if tool_call["name"] == "convert":
        tool_call["args"]["conversion_rate"] = conversion_rate
        tool_message2 = convert.invoke(tool_call)
        messages.append(tool_message2)


result = llm_with_tools.invoke(messages)
print(result.content)
