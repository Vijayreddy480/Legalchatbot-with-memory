#importing neccessary Libraries
from pinecone import Pinecone
import os
from dotenv import load_dotenv
import uuid

#Accessing the APi keys
load_dotenv()
Pinecone_api_key=os.getenv("Pinecone_api_key")
pc=Pinecone(api_key=Pinecone_api_key)

#Creating index for the Chat History If it doesnt exist
index_name="chatbot-history"
if index_name not in pc.list_indexes().names():
    pc.create_index_for_model(
        name=index_name,
        cloud="aws",
        region="us-east-1",
        embed={
            "model":"llama-text-embed-v2",
            "field_map":{"text":"chunk_text"}
        }
    )

history_index=pc.Index("chatbot-history")


