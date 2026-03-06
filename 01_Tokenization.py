import tiktoken
enc=tiktoken.encoding_for_model("gpt-4o")
text="Hey there! My name is Aditya Soran"
tokens=enc.encode(text)
decoded=enc.decode(tokens)
print("Tokens: ",tokens)
print(decoded)