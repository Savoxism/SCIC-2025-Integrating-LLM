import argparse
import os
from tqdm import tqdm
import pandas as pd
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma

# Flow: Clears the DB directory -> loads the CSV file -> splits contents into chunks (if needed) -> adds chunks to Chroma if they’re new

# Usage: python populate_database.py --reset to clear the database

CHROMA_PATH = "chroma"
DATA_PATH = "data/sampled_calculus.csv"  

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)

def load_documents():
    return load_csv_documents(DATA_PATH)

def load_csv_documents(filepath: str) -> list[Document]:
    df = pd.read_csv(filepath)
    documents = []
    for idx, row in df.iterrows():
        content = f"Question: {row['question']}\nAnswer: {row['answer']}"
        # Metadata now uses 'section' and 'id' from the CSV.
        metadata = {"section": row["section"], "id": row["id"]}
        documents.append(Document(page_content=content, metadata=metadata))
    return documents

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate unique IDs for each chunk.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Retrieve existing document IDs.
    existing_items = db.get(include=[]) 
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Filter out documents that are already in the DB.
    new_chunks = []
    for chunk in tqdm(chunks_with_ids, desc="Processing chunks"):
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    print(f"👉 Adding new documents: {len(new_chunks)}")

    if new_chunks:
        batch_size = 1000  
        for i in tqdm(range(0, len(new_chunks), batch_size), desc="Adding batches"):
            batch_chunks = new_chunks[i : i + batch_size]
            batch_ids = [chunk.metadata["id"] for chunk in batch_chunks]
            db.add_documents(batch_chunks, ids=batch_ids)
    else:
        print("✅ No new documents to add")

def calculate_chunk_ids(chunks):
    """
    Create unique IDs for each chunk using the CSV metadata.
    This will create IDs like "curl:29629:0", "curl:29629:1", etc.
    """
    last_doc_id = None
    current_chunk_index = 0

    for chunk in chunks:
        section = chunk.metadata.get("section")
        doc_id = chunk.metadata.get("id")
        current_doc_id = f"{section}:{doc_id}"

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
