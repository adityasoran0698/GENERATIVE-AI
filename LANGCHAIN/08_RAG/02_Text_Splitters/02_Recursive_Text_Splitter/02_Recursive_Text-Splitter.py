from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("JAVA_OOPs.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, chunk_overlap=20, separators=""
)
text = """
What is lorem ipsum, and when did publishers begin using it?
The standard lorem ipsum passage has been a printer's friend for centuries. Like stock photos today, it served as a placeholder for actual content. The original text comes from Cicero's philosophical work "De Finibus Bonorum et Malorum," written in 45 BC.

The use of the lorem ipsum passage dates back to the 1500s. When printing presses required painstaking hand-setting of type, workers needed something to show clients how their pages would look. To save time, they turned to Cicero's words, creating sample books filled with preset paragraphs.

However, it wasn't until the 1960s that the passage became common when Letraset revolutionized the advertising industry with its transfer sheets. These innovative sheets allowed designers to apply pre-printed lorem ipsum text in various fonts and formats directly onto their mockups and prototypes.

What does Lorem Ipsum text say?
Printers in the 1500s scrambled the words from Cicero's "De Finibus Bonorum et Malorum'' after mixing the words in each sentence. The familiar "lorem ipsum dolor sit amet" text emerged when 16th-century printers adapted Cicero's original work, beginning with the phrase "dolor sit amet consectetur."

They abbreviated "dolorem" (meaning "pain") to "lorem," which carries no meaning in Latin. "Ipsum" translates to "itself," and the text frequently includes phrases such as "consectetur adipiscing elit" and "ut labore et dolore." These Latin fragments, derived from Cicero's philosophical treatise, were rearranged to create the standard dummy text that has become a fundamental tool in design and typography across generations.

The short answer is that lorem ipsum text doesn't actually "say" anything meaningful. It's deliberately scrambled Latin that doesn't form coherent sentences. While it comes from Cicero's "De Finibus Bonorum et Malorum," the text has been modified so extensively that it's nonsensical.

Why scrambled text? That's exactly the point. By using text that's unreadable but maintains the general pattern of regular writing — including normal word length, spacing, and punctuation — designers can focus on the visual elements of a layout without the actual content getting in the way. The pseudo-Latin appearance gives it a natural feel while ensuring it won't distract from the design itself.
"""
# result = splitter.split_text(text) -> used to split text
result=splitter.split_documents(docs)
print(result[0].page_content)
print(result[1].page_content)
print(result[2].page_content)
print(result[3].page_content)

