import os
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

def get_embedding():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",  # or "text-embedding-ada-002"
        openai_api_key=os.getenv("OPENAI_API_KEY")  
    )
    return embeddings
