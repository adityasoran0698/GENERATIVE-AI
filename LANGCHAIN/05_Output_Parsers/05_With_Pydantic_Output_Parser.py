from urllib.parse import _ResultMixinStr

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from typing import Optional
from pydantic import BaseModel, Field

from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="openai/gpt-oss-120b", task="text-generation")
model = ChatHuggingFace(llm=llm)


class Report(BaseModel):
    name: str = Field(description="Name of the topic")
    summary: str = Field("Breif summary of the topic")
    word_count: int = Field(description="No. of words in a summary")
    advantages: Optional[list[str]] = Field("Advantages related to the topic if it has")
    disadvantages: Optional[list[str]] = Field(
        "disadvantages related to the topic if it has"
    )

parser=PydanticOutputParser(pydantic_object=Report)
template = PromptTemplate(
    template="Write a detailed prompt on {topic} \n {format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions":parser.get_format_instructions()}
)


chain=template | model | parser
result=chain.invoke({"topic":"SmartPhone"})
print(result)