1. Alpaca Prompting:
### Instructions: <SYSTEM_PROMPT>\n

### Input: <USER_QUERY>\n

### Response:\n

Example:
### Instructions: You are an AI assistant that helps people find information.\n
### Input: What is the capital of France?\n
### Response:\n

2. ChatML Format:

{"role": "system", "content": "<SYSTEM_PROMPT>"}
{"role": "user", "content": "<USER_QUERY>"}
{"role": "assistant", "content": ""}
Example:
{"role": "system", "content": "You are an AI assistant that helps people find information."}
{"role": "user", "content": "What is the capital of France?"}
{"role": "assistant", "content": ""}

3.INST Prompting (Instruction Prompting Style):

[INST] <SYSTEM_PROMPT> [/INST]
<system>\n
<user_query>\n  

Example:
[INST] You are an AI assistant that helps people find information. [/INST]
What is the capital of France?\n

