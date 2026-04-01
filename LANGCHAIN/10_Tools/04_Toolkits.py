from langchain_community.tools import tool


@tool
def add(a: int, b: int) -> int:
    "Adding Two number"
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    "Multiplying Two number"
    return a * b


@tool
def division(a: int, b: int) -> int:
    "Dividing Two number"
    return a / b


@tool
def mod(a: int, b: int) -> int:
    "Finding mod between Two number"
    return a % b


class MathToolkit:
    def get_tools(self):
        return [add, multiply, division, mod]


toolkit = MathToolkit()
tools = toolkit.get_tools()

for tool in tools:
    print(f"{tool.name} => {tool.description}")
