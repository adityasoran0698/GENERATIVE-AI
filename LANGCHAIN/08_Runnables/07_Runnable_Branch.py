from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import (
    RunnableSequence,
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
    RunnableBranch
)

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()
prompt1=PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=['topic']
)
prompt2=PromptTemplate(
    template="Generate a summary of the text \n{text}",
    input_variables=['text']
)
report_gen_chain=RunnableSequence(prompt1,model,parser)
branch_chain=RunnableBranch(
    (lambda x:len(x.split())>500,RunnableSequence(prompt2,model,parser)),
    RunnablePassthrough()
)

final_chain=RunnableSequence(report_gen_chain,branch_chain)
result=final_chain.invoke({"topic":'World war 2'})
print(result)