from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic import hub
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun, tool
import requests
from dotenv import load_dotenv

load_dotenv()

search_tool = DuckDuckGoSearchRun()


@tool
def get_weather(city: str) -> str:
    "This function fetches the weather of the city and returns a json data"
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid=67e87dcb011c26178cb1adde738ec860&units=metric"
    data = requests.get(url)
    return data.json()


llm = ChatOpenAI(model="gpt-4o-mini")
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm=llm, tools=[search_tool, get_weather], prompt=prompt)
agent_executer = AgentExecutor(
    agent=agent, tools=[search_tool, get_weather], verbose=True
)
query = input("You: ")
result = agent_executer.invoke({"input": query})
print(result["output"])

"""
============================================================
       LANGCHAIN CONCEPTS - QUICK NOTES 
============================================================

------------------------------------------------------------
1. HUB (LangChain Hub)
------------------------------------------------------------
- Think of it like an "App Store" but for AI prompts.
- LangChain Hub is an online library where developers share
  ready-made prompts that you can directly download and use.
- In the code:
    prompt = hub.pull("hwchase17/react")
  This line DOWNLOADS a pre-written ReAct prompt from the hub
  made by user "hwchase17", so you don't have to write it yourself.

------------------------------------------------------------
2. ReAct (Reasoning + Acting)
------------------------------------------------------------
- ReAct = "Reasoning" + "Acting" combined together.
- It is a TECHNIQUE (pattern) used to make AI smarter at
  solving problems step by step.
- Normal AI: User asks → AI directly answers.
- ReAct AI:  User asks → AI THINKS → AI ACTS (uses a tool)
             → AI THINKS again → AI ACTS again → Final Answer.

------------------------------------------------------------
3. HOW ReAct WORKS (Step-by-Step)
------------------------------------------------------------
ReAct follows a loop of 3 steps:
 
  STEP 1 → THOUGHT     : AI thinks "What should I do next?"
  STEP 2 → ACTION      : AI uses a tool (e.g., search Google)
  STEP 3 → OBSERVATION : AI reads the tool's result
 
  ...it keeps repeating this loop until it finds the answer.
 
  agent_scratchpad → It's the AI's rough notebook. Every Thought,
  Action, and Observation gets written here so the AI remembers
  what it already did and doesn't repeat steps. Think of it as
  rough paper used while solving — cleared once the final answer
  is ready.
 
Example:
  User: "Who won the 2024 Cricket World Cup?"
 
  Thought    → "I need to search for this."          ┐
  Action     → DuckDuckGo Search ("2024 WC winner")  │ saved in
  Observation→ "India won the 2024 T20 World Cup."   │ scratchpad
  Thought    → "I now have the answer."              ┘
 
  Final Answer → "India won the 2024 T20 Cricket World Cup."

------------------------------------------------------------
4. AGENT
------------------------------------------------------------
- An Agent is the AI's "BRAIN".
- It decides:
    → What to think
    → Which tool to use
    → What to do next
- It reads the prompt (from Hub) and uses the LLM (like GPT-4o)
  to reason through the problem.
- In the code:
    agent = create_react_agent(llm=llm, tools=[search_tool], prompt=prompt)
  This creates a ReAct-style agent that knows HOW to think
  and which tools it can use.

------------------------------------------------------------
5. AGENT EXECUTOR
------------------------------------------------------------
- The Agent Executor is the AI's "BODY" or "MANAGER".
- The Agent (brain) decides what to do.
- The Agent Executor actually DOES it — runs the tools,
  manages the loop, and returns the final answer.
- Think of it like:
    Agent          = Chef (plans the recipe)
    Agent Executor = Kitchen Staff (actually cooks it)

- In the code:
    agent_executer = AgentExecutor(agent=agent, tools=[search_tool], verbose=True)
  "verbose=True" means it will PRINT every thought and action
  so you can see what's happening behind the scenes.

------------------------------------------------------------
QUICK SUMMARY TABLE
------------------------------------------------------------
  Concept          | What it does (Simple)
  -----------------|---------------------------------
  Hub              | Online store for ready-made prompts
  ReAct            | Think → Act → Observe loop pattern
  How ReAct works  | Thought → Action → Observation → Repeat
  Agent            | The brain — decides what to do
  Agent Executor   | The body — actually runs everything
------------------------------------------------------------

============================================================
                        END OF NOTES
============================================================
"""
