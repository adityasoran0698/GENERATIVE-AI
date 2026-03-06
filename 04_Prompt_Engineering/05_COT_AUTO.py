from openai import OpenAI, api_key
from dotenv import load_dotenv
import os
import json

load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("BASE_URL")
)


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
"""
messages_history = [{"role": "system", "content": SYSTEM_PROMPT}]
user_input = input("👉 ")
messages_history.append({"role": "user", "content": user_input})

while True:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages_history,
        response_format={"type": "json_object"},
    )
    raw_data = response.choices[0].message.content
    parsed_data = json.loads(raw_data)
    messages_history.append({"role": "assistant", "content": raw_data})

    if parsed_data.get("step") == "START":
        print("🚀 ", parsed_data.get("answer"))
    if parsed_data.get("step") == "PLAN":
        print("🧠 ", parsed_data.get("answer"))
    if parsed_data.get("step") == "OUTPUT":
        print("✨ ", parsed_data.get("answer"))
        break

"""
Complete Flow of the code:

Start program
↓
Create messages with system rules
↓
Take user input and add to messages
↓
REPEAT:
   Send messages to AI
   AI returns one step (START / PLAN / OUTPUT)
   Print the step
   Save AI reply in messages (memory)
   If step == OUTPUT → STOP
   Else → REPEAT

Example: User asks “What is 2 + 3 * 5?”

Step 1:
The program stores the system rules in messages and then adds the user’s question to messages.

Step 2:
The program sends messages to the AI. The AI replies with the START step saying it understood the problem. This reply is printed and saved back into messages.

Step 3:
The program sends the updated messages to the AI again. The AI replies with a PLAN step explaining how to solve the problem (using BODMAS). This reply is printed and saved into messages.

Step 4:
The program repeats this process and receives more PLAN steps (such as calculating 3 × 5 and then adding 2). Each reply is printed and added to messages as conversation history.

Step 5:
Finally, the AI sends an OUTPUT step with the final answer (17). The program prints the final answer and stops the loop.

Step 6 (Summary):
The program keeps sending the full conversation (messages) to the AI, prints each step in order (START → PLAN → OUTPUT), saves every reply for memory, and stops when the OUTPUT step is received


"""
