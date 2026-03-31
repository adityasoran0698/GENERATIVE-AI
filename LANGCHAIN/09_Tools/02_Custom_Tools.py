from langchain_community.tools import tool

@tool
def multiply(a:int,b:int)->int:
    "Multiplying two number"
    return a*b


result=multiply.invoke({"a":10,"b":3})
print(result)

print(multiply.name)
print(multiply.description)
print(multiply.args)
print(multiply.args_schema.model_json_schema())