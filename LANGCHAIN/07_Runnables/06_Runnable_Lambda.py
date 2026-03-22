from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import (
    RunnableSequence,
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()


def word_counter(text):
    return len(text.split())


prompt1 = PromptTemplate(
    template="Generate a joke on {topic}", input_variables=["topic"]
)
chain1 = RunnableSequence(prompt1, model, parser)
chain2 = RunnableParallel(
    {"joke": RunnablePassthrough(), "words": RunnableLambda(word_counter)}
)
chain3=RunnableSequence(chain1,chain2)
result=chain3.invoke({"topic":"cricket"})
print(result)
