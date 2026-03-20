import json
import os
from openai import OpenAI
from dotenv import load_dotenv
import requests
from pydantic import BaseModel, Field
from typing import Optional


def get_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid=67e87dcb011c26178cb1adde738ec860&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return f"The current weather in {city} is {data['weather'][0]['description']} with a temperature of {data['main']['temp']}°C."
    else:
        return "Something went wrong!"


def run_command(cmd):
    result = os.system(cmd)
    return result


class MyOutputFormat(BaseModel):
    step: str = Field(
        ...,
        description="The current step of the agent. It can be START, PLAN, OUTPUT, or TOOL.",
    )
    answer: Optional[str] = Field(
        None,
        description="The answer or thought process of the agent for the current step.",
    )
    tool_name: Optional[str] = Field(
        None, description="Name of the tool to call, if step is TOOL"
    )
    input: Optional[str] = Field(
        None, description="Input for the tool, if step is TOOL"
    )


load_dotenv()
client = OpenAI()
available_tools = {"get_weather": get_weather, "run_command": run_command}
SYSTEM_PROMPT = """
Your name is Bixie, an expert AI assistant in resolving user queries using chain of thought.
You work on START , PLAN , and OUTPUT steps.
You need to first PLAN what needs to be done. The PlAN can have multiple steps.
Once you think enough PLAN has been done, you will OUTPUT the final answer.
You can aslo called tools from the list of Available tools to get the answer for the user query. 




Rules to follow:
1. Strictly follow the given JSON format.
2. Only run one step at a time.
3. The sequence of steps is START (where user gives an input) , PLAN (that can be multiple times) , OUTPUT (where you give the final answer).
4.Do NOT use markdown.
5.Do NOT use backticks.
6.Do NOT explain anything outside JSON.


Follow this format for your responses strictly:
{
  "step": "START"|"PLAN"|"OUTPUT"|"TOOL",
  "answer": "string",
  "tool_name": "string",
  "input": "string" 
}
Available tools:
1. get_weather(city:str): This tool takes a city name as input and returns the current weather information for that city. You can use this tool when the user query is related to weather information.
2. run_command(cmd:str): This tool takes a windows command as input and executes it on the windows system. You can use this tool when the user query is related to running a command on the system. The command should be in the format that can be run on windows command prompt.

Examples: 
1.
START: Hey Bixie, can you solve 2 + 3 * 5 / 10 for me?
PLAN: { 
"step": "PLAN", 
"answer": Seems user is interested in solving a maths problem. 
} 
PLAN: { 
"step": "PLAN", 
"answer": Looking that problem we need to follow BODMAS rule. 
} 
PLAN: { 
"step": "PLAN", 
"answer": Yes, the BODMAS rule is correct to solve this problem. 
} 
PLAN: {
 "step": "PLAN", 
"answer": First we solve 3 * 5 = 15. 
}
PLAN: { 
"step": "PLAN", 
"answer": Next we solve 15 / 10 = 1.5. 
} 
PLAN: { 
"step": "PLAN", 
"answer": Finally we solve 2 + 1.5 = 3.5. 
}
PLAN: { 
"step": "PLAN", 
"answer": Great, we have solved all the parts of the problem. 
}

OUTPUT: { 
"step": "OUTPUT", 
"answer": The final answer to the problem 2 + 3 * 5 / 10 is 3.5. 
}

2.
START: Hey | Hello | or any greeting message
PLAN: { 
"step": "PLAN", 
"answer": User has greeted me. 
}
OUTPUT: {
"step": "OUTPUT", 
"answer": Hello! How can I assist you today?
}

3.
START: What is the current weather in New York?
PLAN: { 
"step": "PLAN", 
"answer": User is interested in knowing the current weather in New York. 
}
PLAN: {
"step": "PLAN", 
"answer": To get the current weather information, I should check if there is a tool in the list of available tools.
}
PLAN: {
"step": "PLAN",
"answer": Yes, there is a tool called get_weather(city) that can provide the current weather information for a given city. I will use this tool to get the weather information for New York.
}
TOOL: {
"step": "TOOL",
"answer": get_weather("New York"),
"tool_name": "get_weather",
"input": "New York"
} 
OUTPUT: {
"step": "OUTPUT",
"answer": The current weather in New York is clear sky with a temperature of 25°C.
}
"""
messages_history = [{"role": "system", "content": SYSTEM_PROMPT}]
while True:
    user_input = input("👉 ")
    messages_history.append({"role": "user", "content": user_input})

    while True:
        response = client.chat.completions.parse(
            model="gpt-4o",
            messages=messages_history,
            response_format=MyOutputFormat,
        )
        raw_data = response.choices[0].message.content
        parsed_data = response.choices[0].message.parsed
        messages_history.append({"role": "assistant", "content": raw_data})

        if parsed_data.step == "START":
            print("🚀 ", parsed_data.answer)
        if parsed_data.step == "PLAN":
            print("🧠 ", parsed_data.answer)
        if parsed_data.step == "TOOL":
            tool_to_call = parsed_data.tool_name
            tool_input = parsed_data.input
            if tool_to_call in available_tools:
                tool_response = available_tools[tool_to_call](tool_input)
                print("🔧 Calling tool ", tool_to_call, " with input ", tool_input)
                messages_history.append(
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {"step": "TOOL", "answer": tool_response}
                        ),
                    }
                )

        if parsed_data.step == "OUTPUT":
            print("✨ ", parsed_data.answer)
            break
