from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(repo_id="openai/gpt-oss-120b", task="text-generation")
parser=JsonOutputParser()
model = ChatHuggingFace(llm=llm)
template = PromptTemplate(
    template="Write about photo synthesis in detail\n {format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions":parser.get_format_instructions()}
)
chain= template | model | parser
result=chain.invoke({})
print(result)
