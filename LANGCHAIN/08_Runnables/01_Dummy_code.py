import random




class NakliLLM:
    def __init__(self):
        print("LLM created")
    def predict(self, prompt):
        response_list = [
            "AI stands for Artificial Intelligence",
            "Delhi is the capital of India",
            "IPL is cricket league",
        ]
        return {"response": random.choice(response_list)}


class NakliPromptTemplate:
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

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

llm = NakliLLM()
template = NakliPromptTemplate(
    template="Write a {length} report on {topic}", input_variables=["length", "topic"]
)
chain = NakliLLMChain(llm, template)
result = chain.run({"length": "short", "topic": "AI"})
print(result)
