from email.policy import default

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")
parser1 = StrOutputParser()


class feedback(BaseModel):
    sentiment: Literal["Positive", "Negative"] = Field(
        description="Give the sentiment of the feedback"
    )


parser2 = PydanticOutputParser(pydantic_object=feedback)


prompt1 = PromptTemplate(
    template="Classify the sentiment of the following feedback text in to positive or negative: \n {feedback} \n {format_instructions}",
    input_variables=["feedback"],
    partial_variables={"format_instructions": parser2.get_format_instructions()},
)
classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template="Generate a appropriate response to this positive feedback\n {feedback}",
    input_variables=["feedback"],
)
prompt3 = PromptTemplate(
    template="Generate a appropriate response to this negative feedback\n {feedback}",
    input_variables=["feedback"],
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "Positive", prompt2 | model | parser1),
    (lambda x: x.sentiment == "Negative", prompt3 | model | parser1),
    RunnableLambda(lambda x: "Could not find any sentiment"),
)
chain=classifier_chain | branch_chain
result=chain.invoke({"feedback":"This is a wonderful smartphone"})
print(result)

