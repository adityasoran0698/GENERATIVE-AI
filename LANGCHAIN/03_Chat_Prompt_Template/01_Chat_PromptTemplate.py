from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI


# chat_template = ChatPromptTemplate(
#     [
#         SystemMessage(
#             content="You are a helpfull asistant which explains topic about the subject {subject}"
#         ),
#         HumanMessage("Explain about {topic}"),
#     ]
# )

# prompt = chat_template.invoke({"subject": "Science", "topic": "photosynthesis"})

chat_template = ChatPromptTemplate(
    [
        (
            "system",
            "You are a helpfull asistant which explains topic about the subject {subject}",
        ),
        ("human", "Explain about {topic}"),
    ]
)

prompt = chat_template.invoke({"subject": "Science", "topic": "Photosynthesis"})
print(prompt)


"""
LangChain fills prompts dynamically using placeholders (variables) written inside curly braces like {topic} or {subject} in the prompt template.

When the template is created, these placeholders are just empty variables, not real values. 
 
Later, when invoke() is called, we pass a dictionary such as {"subject": "Science", "topic": "Photosynthesis"}. 
 
LangChain then matches the dictionary keys with the placeholders in the template and replaces them with the provided values. 
 
This means the same prompt template can be reused for many different inputs, and each time invoke() runs, LangChain automatically generates a new final prompt by filling the variables with the given data."""

"""
In LangChain, roles like system, human, or ai are understood because each message in ChatPromptTemplate is written as a tuple (role, message). 

LangChain reads the first element of the tuple as the role and uses an internal mapping to convert it into message objects like SystemMessage, HumanMessage, or AIMessage. 

For example, ("system","You are a teacher") becomes a SystemMessage, and ("human","Explain {topic}") becomes a HumanMessage. The text inside the message can contain placeholders like {topic} or {subject}, which LangChain detects as variables and replaces when invoke() is called with values. 

So the flow is: define template → LangChain reads roles from tuples → placeholders are detected → invoke() fills the variables → final prompt messages are generated and sent to the LLM.

"""
