from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_template = ChatPromptTemplate(
    [
        ("system", "you are a helpfull assistant"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "When did i get my refund?"),
    ]
)
chat_history = []

with open("MessageHistory.txt", "r") as f:
    chat_history.extend(f.readlines())


prompt = chat_template.invoke({"chat_history": chat_history})

print(prompt)

"""

In this code, ChatPromptTemplate creates a structured prompt using different message roles like system and human. The MessagesPlaceholder("chat_history") tells LangChain that a list of previous messages will be inserted at this position when the prompt is generated. LangChain reads the tuple format (role, message) to understand who sent the message, and the placeholder acts as a dynamic slot where past conversation messages can be injected. This helps the model remember previous interactions and generate context-aware responses.

The program reads previous messages from MessageHistory.txt and stores them in the chat_history list. When invoke({"chat_history": chat_history}) is called, LangChain replaces the MessagesPlaceholder with the actual chat history messages from the list. This means the final prompt sent to the LLM contains the system instruction → previous conversation history → current user question, allowing the AI to respond based on the entire conversation context rather than just the latest message.
"""
