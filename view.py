from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma

CHROMA_PATH = "chroma"

db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

query = r"""
Find the curl of the vector field $f(x,y,z)\uvec(i) + g(x,y,z)\uvec(j) + h(x,y,z)\uvec(k)$ where $f(x,y,z) = \log (x)$, $g(x,y,z) = e^{z^3}$, and $h(x,y,z) = z^{12}$
"""

sample_docs = db.similarity_search(query, k=5)

for i, doc in enumerate(sample_docs):
    print(f"Document {i+1}:")
    print("ID:", doc.metadata.get("id"))
    print("Section:", doc.metadata.get("section"))
    print("Question and answer:", doc.page_content[:500])
    print("\n" + "="*50 + "\n")
