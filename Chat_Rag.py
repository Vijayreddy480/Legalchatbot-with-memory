#importing necessary libraries
from pinecone import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid
from dotenv import load_dotenv
import os
import fitz
import time
#initializing api keys

load_dotenv()
Pinecone_api_key=os.getenv("Pinecone_api_key")
pc=Pinecone(api_key=Pinecone_api_key)

#extracting text from the pdf
def extract_text_from_pdf(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

#converting data into chunks
def data_into_chunks(data,chunk_size=1500,overlap=300):
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    chunks=splitter.create_documents([data])
    return chunks

#Embedding the chunks using fast api inbuilt embed function
#creating the vectorDatabase
index_name="minor-project"
if index_name not in pc.list_indexes().names():
    pc.create_index_for_model(
        name=index_name,
        cloud="aws",
        region="us-east-1",
        embed={
            "model": "llama-text-embed-v2",
            "field_map": {"text": "chunk_text"}
        }
    )

index=pc.Index("minor-project")
desc = index.describe_index_stats(namespace="default")
vector_count = desc.get("namespaces", {}).get("default", {}).get("vector_count", 0)
if vector_count == 0:
    #uploading the pdf file
    def extract_text_from_pdfs(pdf_name):
        text = extract_text_from_pdf(pdf_name)
        chunks = data_into_chunks(text)
        # Prepare the chunk texts
        chunk_texts = [chunk.page_content for chunk in chunks]
        #creating index and it chunks 
        records = [
            {
                "_id": str(uuid.uuid4()),
                "chunk_text": chunk,
                "document_name":pdf_name
            }
            for chunk in chunk_texts
        ]
        batch_size = 96
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            #uploading into vector database
            index.upsert_records(records=batch,namespace="default")
            time.sleep(30)
    for i in ["constitution.pdf","IPC_186045.pdf"]:
        extract_text_from_pdfs(i)


