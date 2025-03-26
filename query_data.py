import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from get_embedding_function import get_embedding_function

from google import genai
from google.genai import types

from dotenv import load_dotenv
import os

# example: python3 query_data.py "Find the curl of the vector field $f(x,y,z)\uvec{i} + g(x,y,z)\uvec{j} + h(x,y,z)\uvec{k}$ where $f(x,y,z) = \log (x)$, $g(x,y,z) = e^{z^3}$, and $h(x,y,z) = z^{12}$"

load_dotenv()
genai_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

MODEL_NAME = "gemini-2.0-flash"
CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Please solve the following question step by step. First, provide a detailed explanation of your reasoning (chain-of-thought) before arriving at the final answer. Then, clearly state the final answer on a new line prefixed by "Final Answer:".

Context:
{context}

Question:
{question}

Chain of Thought (detailed explanation):
"""

def main():
    parser = argparse.ArgumentParser(description="Query the vector DB and get an answer from Gemini 2.0 Flash.")
    parser.add_argument("query_text", type=str, help="The question you want to ask.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)

def query_rag(query_text: str):
    # Initialize the embedding function and load the Chroma vector DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Perform similarity search in the DB to retrieve the top 5 related documents.
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    
    # Build the full prompt by inserting the context and the question.
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Call Gemini 2.0 Flash using streaming to get the answer with chain-of-thought.
    model = MODEL_NAME
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        response_mime_type="text/plain",
    )

    print("Response:", end=" ")
    response_text = ""
    for chunk in genai_client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if chunk.text:
            print(chunk.text, end="", flush=True)
            response_text += chunk.text

    # For traceability, print out the document IDs of the context.
    sources = [doc.metadata.get("id", None) for doc, _ in results]
    print(f"\nSources: {sources}")
    return response_text

if __name__ == "__main__":
    main()


