from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()
prompt1 = PromptTemplate(
    template="Generate short notes on the following text: \n {text} ",
    input_variables=["text"],
)

prompt2 = PromptTemplate(
    template="Generate 5 short questions for quiz on the following text: \n {text}",
    input_variables=["text"],
)

prompt3 = PromptTemplate(
    template="Merge the following notes and quiz question in to single document\n Notes-> {notes} \n Quiz->{quiz}",
    input_variables=["notes", "quiz"],
)

model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {"notes": prompt1 | model | parser, "quiz": prompt2 | model | parser}
)

merge_chain = prompt3 | model | parser

chain = parallel_chain | merge_chain
text = """
WHAT IS AN OUTPUT PARSER?
--------------------------
When you ask an LLM (like GPT or HuggingFace model) a question,
it always returns a plain text string as response.

But in real applications, you need data in a proper format like:
  - A Python dictionary
  - A JSON object
  - A list
  - A custom structure

Output Parsers help you convert the LLM's plain text response
into a structured, usable format automatically.

Simple flow:
  LLM Response (plain text)  -->  Output Parser  -->  Structured Data


================================================================
        TYPES OF OUTPUT PARSERS IN LANGCHAIN
================================================================

----------------------------------------------------------------
TYPE 1: String Output Parser
----------------------------------------------------------------

WHAT IT DOES:
  The simplest parser. It just takes the LLM's output and
  returns it as a plain Python string. No formatting at all.

EXAMPLE:
  from langchain_core.output_parsers import StrOutputParser

  parser = StrOutputParser()
  result = parser.invoke(llm_response)
  # result = "Paris is the capital of France."

WHEN TO USE:
  When you just need raw text output and don't care about
  structure. Good for simple Q&A or summarization tasks.

LIMITATION:
  You get a raw string. If you want specific fields like
  "city name" or "population" separately, you can't extract
  them easily. You would have to manually split/parse the string
  yourself, which is error-prone.

  --> TO OVERCOME THIS: Use JSON Output Parser


----------------------------------------------------------------
TYPE 2: JSON Output Parser
----------------------------------------------------------------

WHAT IT DOES:
  Tells the LLM to respond in JSON format, then automatically
  parses that JSON string into a Python dictionary.

EXAMPLE:
  from langchain_core.output_parsers import JsonOutputParser

  parser = JsonOutputParser()
  # LLM responds: '{"name": "Paris", "country": "France"}'
  result = parser.invoke(llm_response)
  # result = {"name": "Paris", "country": "France"}

WHEN TO USE:
  When you want key-value structured data from the LLM response
  and you are okay with a flexible structure.

LIMITATION:
  You don't define or enforce a schema exist in the JSON.
  The LLM might return different keys each time, or miss a field
  you expected. There is no validation of the structure.

  For example, sometimes LLM may return:
    {"city": "Paris"}      <-- used "city" instead of "name"
    {"name": "Paris", "population": 2000000}  <-- extra field

  --> TO OVERCOME THIS: Use Structured Output Parser


----------------------------------------------------------------
TYPE 3: Structured Output Parser
----------------------------------------------------------------

WHAT IT DOES:
  You define exactly what type of schema or fields you want in the response
  using ResponseSchema. The parser then instructs the LLM to
  return those exact fields and parses the result accordingly.

EXAMPLE:
  from langchain.output_parsers import StructuredOutputParser, ResponseSchema

  schemas = [
      ResponseSchema(name="name", description="Name of the city"),
      ResponseSchema(name="country", description="Country of the city"),
      ResponseSchema(name="population", description="Population of the city")
  ]

  parser = StructuredOutputParser.from_response_schemas(schemas)
  format_instructions = parser.get_format_instructions()
  # This gives instructions to LLM to follow the schema

  result = parser.parse(llm_response)
  # result = {"name": "Paris", "country": "France", "population": "2.1 million"}

WHEN TO USE:
  When you know exactly what fields you want and need consistency
  across multiple LLM calls.

LIMITATION:
  Fields are just strings. You cannot enforce data types.
  For example, you can't force "population" to be an integer.
  The LLM might return "2.1 million" (string) instead of 2100000
  (integer). No type validation happens.

Schema says:  name="population", description="Population of the city"
                        ↓
 LLM can return ANYTHING:
  "population": "2.1 million"      ← string
  "population": "approximately 2M" ← string
  "population": 2100000            ← integer (lucky!)
  "population": "unknown"          ← random string
 
There is No validation of data in the Output.

  --> TO OVERCOME THIS: Use Pydantic Output Parser


----------------------------------------------------------------
TYPE 4: Pydantic Output Parser
----------------------------------------------------------------

WHAT IT DOES:
  You define a Pydantic model (a Python class with typed fields),
  and the parser forces the LLM to return data matching that
  exact model — with proper data types and validation.

EXAMPLE:
  from langchain_core.output_parsers import PydanticOutputParser
  from pydantic import BaseModel

  class CityInfo(BaseModel):
      name: str
      country: str
      population: int   # strictly integer!

  parser = PydanticOutputParser(pydantic_object=CityInfo)
  result = parser.invoke(llm_response)

  # result is a CityInfo object:
  # result.name       = "Paris"
  # result.country    = "France"
  # result.population = 2100000  (actual integer, not string)

WHEN TO USE:
  When you need strict data types and full validation.
  Best choice for production applications where data quality
  matters. Works with both old and new langchain versions.

================================================================
        QUICK COMPARISON TABLE
================================================================

  Parser               | Returns       | Type Safe | Validation
  ---------------------|---------------|-----------|------------
  StrOutputParser      | Plain string  | No        | None
  JsonOutputParser     | Dict (any)    | No        | None
  StructuredOutput     | Dict (fixed)  | No        | Field names
  PydanticOutput       | Pydantic obj  | Yes       | Full

================================================================
        RECOMMENDED PROGRESSION
================================================================

  Beginner / Simple tasks
        |
        v
  StrOutputParser  (just text)
        |
        v
  JsonOutputParser  (need key-value data)
        |
        v
  StructuredOutputParser  (need specific fields)
        |
        v
  PydanticOutputParser  (need specific fields + validation)


"""
result = chain.invoke({"text": text})
print(result)