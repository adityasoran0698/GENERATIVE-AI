from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()
# loads API key from .env file

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("BASE_URL")
)
# creates AI client


# Chain of Thought Prompting
SYSTEM_PROMPT = """
Your name is Bixie, an exper AI assistant in resolving user queries using chain of thought.
You work on START , PLAN , and OUTPUT steps.
You need to first PLAN what needs to be done. The PlAN can have multiple steps.
Once you think enough PLAN has been done, you will OUTPUT the final answer.

Rules to follow:
1. Strictly follow the given JSON format.
2. Only run one step at a time.
3. The sequence of steps is START (where user gives an input) , PLAN (that can be multiple times) , OUTPUT (where you give the final answer).

OUTPUT JSON FORMAT:
{
  "step": "START"|"PLAN"|"OUTPUT",
  "answer": "string"
}

Examples: 
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
"""
# This system prompt teaches the AI:
# - to think step by step (PLAN)
# - and answer only in structured JSON
# This is Chain of Thought (CoT) prompting


response = client.chat.completions.create(
    model="gpt-4o-mini",
    response_format={"type": "json_object"},
    # forces AI to reply only in JSON
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        # sets the thinking rules for AI
        {
            "role": "user",
            "content": "Write a code to print sum of n natural numbers in java",
        },
        # user question
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "step": "PLAN",
                    "answer": "User wants a java code to print sum of n natural numbers.",
                }
            ),
        },
        # MANUALLY added reasoning step
        # you are telling the AI: "this is how you should think"
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "step": "PLAN",
                    "answer": "To write the code, we will need a method that takes an integer n as input.",
                }
            ),
        },
        # another MANUAL reasoning step
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "step": "PLAN",
                    "answer": "Next, we can use a loop to calculate the sum of the first n natural numbers.",
                }
            ),
        },
        # another MANUAL reasoning step
        {
            "role": "assistant",
            "content": json.dumps(
                {"step": "PLAN", "answer": "We will then print the calculated sum."}
            ),
        },
        # final MANUAL planning step
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "step": "OUTPUT",
                    "answer": "Here is the Java code to print the sum of n natural numbers: ...",
                }
            ),
        },
        # final answer step
    ],
)

print(response.choices[0].message.content)
# prints the final structured response
