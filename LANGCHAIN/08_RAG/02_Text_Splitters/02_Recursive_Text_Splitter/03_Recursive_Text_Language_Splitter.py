from langchain_classic.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("JAVA_OOPs.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=200,
    chunk_overlap=20,
)
text = """
# Define a class
class Student:
    # Constructor (runs when object is created)
    def __init__(self, name, age):
        self.name = name
        self.age = age

    # Function inside class
    def display_info(self):
        print("Student Name:", self.name)
        print("Student Age:", self.age)


# Create (invoke) the class -> object creation
student1 = Student("Aditya", 22)

# Call the function of the class
student1.display_info()
"""
# result = splitter.split_text(text) -> used to split text
result = splitter.split_text(text)
print(result[0])
print(result[1])
print(result[2])


"""
👉 RecursiveCharacterTextSplitter.fromLanguage
🔹 What is fromLanguage?

It is a special method of RecursiveCharacterTextSplitter that splits text based on programming language syntax instead of normal separators.

👉 It uses language-specific rules (like functions, classes, etc.)

🔹 Why use it?

Normal splitter uses:

paragraph → line → word → character

But in code files, this can break logic ❌

👉 So fromLanguage:

Keeps code structure safe
Splits based on logical blocks (functions, classes, etc.)
🔹 How it Works

Instead of:

["\n\n", "\n", " ", ""]

It uses language-aware separators, for example (Python):

1. class
2. def
3. \n\n
4. \n
5. (words)
6. (character)

👉 Process:

Try splitting by class
If too big → split by function (def)
If still big → fallback to normal splitting
Continue until chunk size is satisfied
"""
