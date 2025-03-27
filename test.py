from get_embedding import get_embedding
from langchain_chroma import Chroma

CHROMA_PATH = "chroma"

db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding)

query = r"""
How many vertical asymptotes does the graph of y=2/(x^2+x-6) have?
"""

sample_docs = db.similarity_search(query, k=5)

for i, doc in enumerate(sample_docs):
    print(f"Document {i+1}:")
    print("ID:", doc.metadata.get("id"))
    print("Section:", doc.metadata.get("type"))
    print("Question and answer:", doc.page_content[:500])
    print("\n" + "="*50 + "\n")
