from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="openai/gpt-oss-120b", task="text-generation")
model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name="fact 1", description="fact 1 about the topic"),
    ResponseSchema(name="fact 2", description="fact 2 about the topic"),
    ResponseSchema(name="fact 3", description="fact 3 about the topic"),
    ResponseSchema(name="fact 4", description="fact 4 about the topic"),
    ResponseSchema(name="fact 5", description="fact 5 about the topic"),
]
parser = StructuredOutputParser.from_response_schemas(schema)
template = PromptTemplate(
    template="Write 5 facts about {topic} \n {format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain= template | model | parser
result=chain.invoke({"topic":"Blackhole"})
print(result)
