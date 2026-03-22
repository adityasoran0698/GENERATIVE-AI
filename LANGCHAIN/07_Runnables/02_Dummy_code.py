import random
from abc import ABC, abstractmethod


class Runnable(ABC):
    @abstractmethod
    def invoke(input_data):
        pass


class NakliLLM(Runnable):
    def __init__(self):
        print("LLM created")

    def invoke(self, prompt):
        response_list = [
            "AI stands for Artificial Intelligence",
            "Delhi is the capital of India",
            "IPL is cricket league",
        ]
        return {"response": random.choice(response_list)}

    def predict(self, prompt):
        response_list = [
            "AI stands for Artificial Intelligence",
            "Delhi is the capital of India",
            "IPL is cricket league",
        ]
        return {"response": random.choice(response_list)}


class NakliPromptTemplate(Runnable):
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def invoke(self, input_dict):
        return self.template.format(**input_dict)

    def format(self, input_dict):
        return self.template.format(**input_dict)


class NakliLLMChain:
    def __init__(self, llm, prompt):
        self.llm = llm
        self.prompt = prompt

    def run(self, input_dict):
        final_prompt = self.prompt.format(input_dict)
        result = self.llm.predict(final_prompt)
        return result["response"]


class RunnableConnector(Runnable):
    def __init__(self, runnable_list):
        self.runnable_list = runnable_list

    def invoke(self, input_data):
        for runnable in self.runnable_list:
            input_data = runnable.invoke(input_data)
        return input_data


class NakliStrOutputParser(Runnable):
    def __init__(self):
        pass

    def invoke(self, input_data):
        return input_data["response"]


llm = NakliLLM()
template = NakliPromptTemplate(
    template="Write a {length} report on {topic}", input_variables=["length", "topic"]
)


# prompt = template.format({"length": "short", "topic": "AI"})

# print(prompt)
# result = llm.predict(prompt)
# print(result["response"])
# chain = NakliLLMChain(llm, template)
# result = chain.run({"length": "short", "topic": "AI"})
# print(result)

chain = RunnableConnector([template, llm])
result = chain.invoke({"length": "short", "topic": "AI"})
parser = NakliStrOutputParser()
final_result = parser.invoke(result)
print(result)
print(final_result)

# Implementing Runnable
template2 = NakliPromptTemplate(
    template="Write a joke about {topic}", input_variables=["topic"]
)
template3 = NakliPromptTemplate(
    template="Explain the joke {response}", input_variables=["response"]
)
llm = NakliLLM()
parser = NakliStrOutputParser()

chain1 = RunnableConnector([template2, llm])
chain1.invoke({"topic": "cricket"})
chain2 = RunnableConnector([template3, llm])
result2 = chain2.invoke({"response": "this is the joke"})
final_chain = RunnableConnector([chain1, chain2])
result3 = final_chain.invoke({"topic": "cricket"})
print(result2)
print(result3)
