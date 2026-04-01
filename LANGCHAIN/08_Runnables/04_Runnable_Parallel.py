from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Generate a tweet about {topic}", input_variables=["topic"]
)
prompt2 = PromptTemplate(
    template="Generate a linkedin post about {topic}", input_variables=["topic"]
)

prompt3 = PromptTemplate(
    template="Merge the following tweet and linkedin post in a single document by seperating them with headings. \n tweet -> {tweet}\n linkedin_post->{linkedin}",
    input_variables=["tweet", "linkedin"],
)

parallel_chain = RunnableParallel(
    {
        "tweet": RunnableSequence(prompt1, model, parser),
        "linkedin":RunnableSequence(prompt2,model,parser)
    }
)
sequential_chain= RunnableSequence(prompt3,model,parser)
chain=RunnableSequence(parallel_chain,sequential_chain)
result=chain.invoke({"topic":"Artificial Intelligence"})
print(result)