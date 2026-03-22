from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()
prompt1 = PromptTemplate(
    template="Tell me a joke on {topic}", input_variables=["topic"]
)
prompt2 = PromptTemplate(template="Explain this joke {joke} and merge the joke and its explanation in single document to display", input_variables=["joke"])

chain1 = RunnableSequence(prompt1, model, parser)
chain2=RunnableSequence(prompt2,model,parser)
final_chain=RunnableSequence(chain1,chain2)
result=final_chain.invoke({"topic":"sports"})
print(result)
