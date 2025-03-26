import argparse
import os
import shutil
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma

# Flow: Clears the DB directory -> loads the CSV file -> splits contents into chunks (if needed) -> adds chunks to Chroma if they’re new
# Usage: python populate_database.py --reset to clear the database

CHROMA_PATH = "chroma"
DATA_PATH = "data/algebra.csv"  # CSV with columns: problem, level, type, solution

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Clear the database before populating.")
    args = parser.parse_args()
    
    if args.reset:
        print("Clearing database…")
        clear_database()

    documents = load_csv_documents(DATA_PATH)
    chunks = split_documents(documents)
    add_to_chroma(chunks)

def load_csv_documents(filepath: str) -> list[Document]:
    df = pd.read_csv(filepath)
    documents = []
    for idx, row in df.iterrows():
        content = f"Problem: {row['problem']}\nSolution: {row['solution']}"
        metadata = {
            "type": row["type"],
            "level": row["level"],
            "orig_id": str(idx)
        }
        documents.append(Document(page_content=content, metadata=metadata))
    return documents

def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])  
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if new_chunks:
        print(f"👉 Adding new documents: {len(new_chunks)}")
        batch_size = 1000  # Process in batches to avoid exceeding maximum batch sizes.
        for i in range(0, len(new_chunks), batch_size):
            batch_chunks = new_chunks[i:i+batch_size]
            batch_ids = [chunk.metadata["id"] for chunk in batch_chunks]
            db.add_documents(batch_chunks, ids=batch_ids)
    else:
        print("✅ No new documents to add")

def calculate_chunk_ids(chunks):
    last_doc_id = None
    current_chunk_index = 0

    for chunk in chunks:
        doc_type = chunk.metadata.get("type", "Unknown")
        level = chunk.metadata.get("level", "Unknown")
        orig_id = chunk.metadata.get("orig_id", "0")
        current_doc_id = f"{doc_type}:{level}:{orig_id}"
        
        # If it's the same document as the previous chunk, increment the chunk index.
        if current_doc_id == last_doc_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        
        new_chunk_id = f"{current_doc_id}:{current_chunk_index}"
        last_doc_id = current_doc_id
        chunk.metadata["id"] = new_chunk_id

    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":
    main()