from langchain_openai import ChatOpenAI
from typing import Optional, TypedDict, Annotated
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini")


class Review(TypedDict):
    key_themes: Annotated[
        list[str], "Write down all the key themes discussed in the review in a list"
    ]
    summary: Annotated[str, "A breif summary of the review"]
    sentiments: Annotated[
        str,
        "Write down the sentiments of the review. For ex- Positive,Negative or Neutral",
    ]
    topic: Annotated[str, "Write down the topic related to the review"]
    pros: Annotated[
        Optional[list[str]],
        "Write down all the pros if exist in the review inside a list",
    ]
    cons: Annotated[
        Optional[list[str]],
        "Write down all the cons if exist in the review inside a list",
    ]
    name: Annotated[Optional[str], "Write down the name of the reviewer"]


structured_model = model.with_structured_output(Review)

result = structured_model.invoke(
    "I’ve been using this smartphone for about a month now, and overall, it offers a solid experience with a few noticeable drawbacks. On the positive side, the performance is fast and reliable — apps open quickly, multitasking is smooth, and the display is vibrant with excellent color accuracy. The battery life is impressive and comfortably lasts a full day with heavy usage. The design also feels premium and comfortable to hold. However, there are some downsides. The camera performs well in good lighting but struggles in low-light conditions, producing grainy images. The device also heats up slightly during gaming or prolonged use. Additionally, the price feels a bit high compared to competitors offering similar features. Overall, it’s a good device, but not without its flaws.-Rahul Mehta"
)
print(result)
