from langchain_classic.schema import Document

doc1 = Document(
    page_content="Virat Kohli is one of the best batsmen in the world. He has scored many centuries in international cricket and is known for his aggressive playing style.",
    metadata={"player": "Virat Kohli", "role": "Batsman", "id": 1},
)

doc2 = Document(
    page_content="Rohit Sharma is the captain of the Indian cricket team in limited overs. He is famous for hitting big sixes and has multiple double centuries in ODI cricket.",
    metadata={"player": "Rohit Sharma", "role": "Batsman", "id": 2},
)

doc3 = Document(
    page_content="MS Dhoni is one of the most successful captains of India. He is known for his calm nature and finishing matches under pressure.",
    metadata={"player": "MS Dhoni", "role": "Wicketkeeper", "id": 3},
)

doc4 = Document(
    page_content="Jasprit Bumrah is a fast bowler known for his unique bowling action and deadly yorkers. He plays a key role in India's bowling attack.",
    metadata={"player": "Jasprit Bumrah", "role": "Bowler", "id": 4},
)

doc5 = Document(
    page_content="Hardik Pandya is an all-rounder who contributes with both bat and ball. He is known for his power hitting and fast bowling.",
    metadata={"player": "Hardik Pandya", "role": "All-rounder", "id": 5},
)

docs = [doc1, doc2, doc3, doc4, doc5]
print(docs)