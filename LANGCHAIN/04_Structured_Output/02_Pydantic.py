from pydantic import BaseModel, Field
from typing import Optional
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# DEMO...
# class Student(BaseModel):
#     name:str
#     age:int
#     email:EmailStr
#     cgpa:float=Field(gt=0,lt=10)

# new_student={"name":"Aditya","age":22,"email":"abc@gmail.com","cgpa":"5"}
# result=Student(**new_student)
# json=result.model_dump_json()
# print(result)
# print(json)


model = ChatOpenAI(model="gpt-4o-mini")


class Review(BaseModel):
    key_themes: list[str] = Field(
        description="Write down all the key themes discussed in the review in a list"
    )
    summary: str = Field(description="A breif summary of the review")
    sentiments: str = Field(
        description="Write down the sentiments of the review. For ex- Positive,Negative or Neutral"
    )
    topic: str = Field(description="Write down the topic related to the review")
    pros: Optional[list[str]] = Field(
        description="Write down all the pros if exist in the review inside a list"
    )
    cons: Optional[list[str]] = Field(
        description="Write down all the cons if exist in the review inside a list"
    )

    name: Optional[str] = Field(description="Write down the name of the reviewer")


structured_model = model.with_structured_output(Review)

result = structured_model.invoke(
    "I’ve been using this smartphone for about a month now, and overall, it offers a solid experience with a few noticeable drawbacks. On the positive side, the performance is fast and reliable — apps open quickly, multitasking is smooth, and the display is vibrant with excellent color accuracy. The battery life is impressive and comfortably lasts a full day with heavy usage. The design also feels premium and comfortable to hold. However, there are some downsides. The camera performs well in good lighting but struggles in low-light conditions, producing grainy images. The device also heats up slightly during gaming or prolonged use. Additionally, the price feels a bit high compared to competitors offering similar features. Overall, it’s a good device, but not without its flaws.-Rahul Mehta"
)
json = result.model_dump_json()
print(json)
