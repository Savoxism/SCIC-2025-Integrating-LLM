from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()

def get_embedding_function():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",  # or "text-embedding-ada-002"
        openai_api_key=os.getenv("OPENAI_API_KEY")  
    )
    return embeddings
