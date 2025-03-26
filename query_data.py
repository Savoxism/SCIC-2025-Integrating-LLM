# query_data.py
import argparse
import os
import base64
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from get_embedding_function import get_embedding_function
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
genai_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# python3 query_data.py "How many vertical asymptotes does the graph of y=2/(x^2+x-6) have?"

MODEL_NAME = "gemini-2.0-flash"
CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are provided with the following context from related documents.
Even if the question is not explicitly clear, use the context to generate a plausible and detailed answer.
Before giving your final answer, provide a detailed chain-of-thought reasoning process. You have to provide the explanation, then you will provide the final answer.

Context:
{context}

Question:
{question}

Chain of Thought: Firstly I need to ...
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)

def query_rag(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    model = MODEL_NAME
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
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

    sources = [doc.metadata.get("id", "Unknown") for doc, _ in results]
    print(f"\nSources: {sources}")
    return response_text

if __name__ == "__main__":
    main()
