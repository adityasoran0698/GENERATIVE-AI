from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import (
    RunnableSequence,
    RunnableParallel,
    RunnablePassthrough,
)

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()
prompt1 = PromptTemplate(
    template="Generate a joke on {topic}", input_variables=["topic"]
)
prompt2 = PromptTemplate(template="Explain the joke.\n{joke}", input_variables=["joke"])
prompt3 = PromptTemplate(
    template="Merge the joke and its explanation in to a single document.\nJoke->{joke}\n explanation->{explanation}",
    input_variables=["joke", "explanation"],
)
joke_chain = RunnableSequence(prompt1, model, parser)
parallel_chain = RunnableParallel(
    {
        "joke": RunnablePassthrough(),
        "explanation": RunnableSequence(prompt2, model, parser),
    }
)

merge_chain = RunnableSequence(prompt3, model, parser)

chain = RunnableSequence(joke_chain, parallel_chain, merge_chain)
result=chain.invoke({"topic":"Banana"})
print(result)
