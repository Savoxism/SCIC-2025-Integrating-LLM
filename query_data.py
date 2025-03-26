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
Before giving your final answer, provide a detailed chain-of-thought reasoning process.
Finally, state the final answer on a new line prefixed by "Final Answer:".

Context:
{context}

Question:
{question}

Chain of Thought:
"""

def is_image_file(filename: str) -> bool:
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))

def get_text_from_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        image_data = f.read()
    image_b64 = base64.b64encode(image_data).decode("utf-8")
    prompt = f"Describe the content of the image concisely:\n[Image (base64): {image_b64}]"
    
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
    config = types.GenerateContentConfig(
        temperature=0.0, top_p=1, top_k=1, max_output_tokens=100, response_mime_type="text/plain"
    )
    
    description = ""
    for chunk in genai_client.models.generate_content_stream(model=MODEL_NAME, contents=contents, config=config):
        if chunk.text:
            description += chunk.text.strip()
    return description

def main():
    parser = argparse.ArgumentParser(
        description="Query the vector DB using a text or image input and get an answer from Gemini 2.0 Flash."
    )
    parser.add_argument("input", type=str, help="A text query or a path to an image file.")
    args = parser.parse_args()
    query_rag(args.input)

def query_rag(query_input: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    if is_image_file(query_input):
        print(f"Processing image: {query_input}")
        extracted_text = get_text_from_image(query_input)
        query_text = f"[Image-derived Query] {extracted_text}. Please provide a detailed answer."
    else:
        print("Processing text query.")
        query_text = query_input

    results = db.similarity_search_with_score(query_text, k=5)
    context = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(context=context, question=query_text)
    
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
    config = types.GenerateContentConfig(
        temperature=1, top_p=0.95, top_k=40, max_output_tokens=8192, response_mime_type="text/plain"
    )

    print("Response:", end=" ")
    response_text = ""
    for chunk in genai_client.models.generate_content_stream(model=MODEL_NAME, contents=contents, config=config):
        if chunk.text:
            print(chunk.text, end="", flush=True)
            response_text += chunk.text

    sources = [doc.metadata.get("id") for doc, _ in results]
    print(f"\nSources: {sources}")
    return response_text

if __name__ == "__main__":
    main()
