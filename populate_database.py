import argparse
import os
import shutil
from tqdm import tqdm
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

    documents = load_documents(DATA_PATH)
    chunks = split_documents(documents)
    add_to_chroma(chunks)

def load_documents(filepath: str) -> list[Document]:
    df = pd.read_csv(filepath)
    documents = []
    for idx, row in df.iterrows():
        content = f"Problem: {row['problem']}\nSolution: {row['solution']}"
        metadata = {"level": row["level"], "type": row["type"], "id": str(idx)}
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
    chunks = calculate_chunk_ids(chunks)
    
    # Grab existing IDs from the DB to avoid duplicates.
    existing_ids = set(db.get(include=[])["ids"])
    print(f"Existing docs in DB: {len(existing_ids)}")
    
    new_chunks = [chunk for chunk in tqdm(chunks, desc="Processing chunks")
                  if chunk.metadata["id"] not in existing_ids]
    print(f"Adding {len(new_chunks)} new documents.")
    
    if new_chunks:
        batch_size = 1000
        for i in tqdm(range(0, len(new_chunks), batch_size), desc="Adding batches"):
            batch = new_chunks[i : i + batch_size]
            batch_ids = [chunk.metadata["id"] for chunk in batch]
            db.add_documents(batch, ids=batch_ids)
    else:
        print("No new documents to add.")

def calculate_chunk_ids(chunks: list[Document]) -> list[Document]:
    last_doc_id = None
    chunk_index = 0
    for chunk in chunks:
        doc_id = chunk.metadata.get("id")
        if doc_id == last_doc_id:
            chunk_index += 1
        else:
            chunk_index = 0
            last_doc_id = doc_id
        chunk.metadata["id"] = f"{doc_id}:{chunk_index}"
    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("Database cleared.")

if __name__ == "__main__":
    main()
