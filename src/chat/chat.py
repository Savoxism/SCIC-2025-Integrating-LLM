# query_data.py
import argparse
import os

from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from google import genai
from google.genai import types

from utils.get_embedding import get_embedding
from database.database import retrieve_docs

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

ChromaDB = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding)

def get_response(question: str, image_path: str=None):
    """
    Tạo câu trả lời dựa trên truy vấn đầu vào bằng cách sử dụng mô hình AI và dữ liệu truy xuất. 

    Hàm này thực hiện các bước sau:
    1. Truy xuất các tài liệu liên quan đến câu hỏi (`question`) bằng cách sử dụng `retrieve_docs()`.
    2. Tạo context từ các tài liệu truy xuất để cung cấp thêm thông tin cho mô hình.
    3. Xây dựng prompt bằng cách sử dụng `ChatPromptTemplate`.
    4. Tương tác với Google Generative AI để tạo câu trả lời.
    5. Nếu có ảnh (`image_path`), tải ảnh lên và sử dụng làm input bổ sung.
    6. Nhận phản hồi từ mô hình AI theo dạng streaming và ghép lại thành văn bản hoàn chỉnh.

    Args:
        question (str): Câu hỏi hoặc truy vấn của người dùng.
        image_path (str, optional): Đường dẫn đến ảnh (nếu có) để cung cấp thêm thông tin cho mô hình. Mặc định là `None`.

    Returns:
        str: Câu trả lời được tạo bởi mô hình AI dựa trên context và truy vấn đầu vào."
    """
    
    results = retrieve_docs(question, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=question)

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    parts = [types.Part.from_text(text=prompt)]

    if image_path:
        file_ref = client.files.upload(file=image_path)
        parts.append(types.Part.from_uri(file_uri=file_ref.uri, mime_type=file_ref.mime_type,))

    contents = [
        types.Content(
            role="user",
            parts=parts,
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
        model=MODEL_NAME,
        contents=contents,
        config=generate_content_config,
    ):
        if chunk.text:
            # print(chunk.text, end="", flush=True)
            response_text += chunk.text

    # sources = [doc.metadata.get("id", "Unknown") for doc, _ in results]
    # print(f"\nSources: {sources}")

    return response_text

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    get_response(query_text)

if __name__ == "__main__":
    main()