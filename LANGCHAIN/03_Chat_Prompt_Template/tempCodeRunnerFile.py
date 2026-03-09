from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# chat_template = ChatPromptTemplate(
#     [
#         ("system", "you are a helpfull assistant"),
#         MessagesPlaceholder(variable_name="chat_history")(
#             "human", "When did i get my refund?"
#         ),
#     ]
# )