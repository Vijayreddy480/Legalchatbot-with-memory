# importing necessary Libraries
from mem0 import Memory
from dotenv import load_dotenv
import os

#initializing the dotenv file
load_dotenv()
#custom update delete prompt
UPDATE_MEMORY_PROMPT = """
You are a smart memory manager which controls the memory of a legal assistance system.
You can perform four operations: (1) add into the memory, (2) update the memory, (3) delete from the memory, and (4) no change.

The memory elements are structured objects and can include:
- text: factual statements from the user
- metadata: structured fields such as { "category": "...", "case_status": "..." }

For every retrieved fact or metadata update, compare with the existing memory and decide whether to:
- ADD: Add it as a new memory element
- UPDATE: Update an existing memory element (keep the same ID)
- DELETE: Delete an existing memory element (contradiction or removal)
- NONE: No change needed (already present or irrelevant)

When updating metadata, ensure you only change the field that differs, while keeping the same `id`.

---

## Examples with Metadata

**Example 1 - Adding New Case Metadata**
- Old Memory:
    [
        { "id": "0", "text": "User had a criminal case for theft", "metadata": { "category": "Criminal", "case_status": "Closed" } }
    ]
- Retrieved facts: ["User is consulting a lawyer for a property dispute"]
- Retrieved metadata: { "category": "Property", "case_status": "Active" }
- New Memory:
    {
        "memory": [
            { "id": "0", "text": "User had a criminal case for theft", "metadata": { "category": "Criminal", "case_status": "Closed" }, "event": "NONE" },
            { "id": "1", "text": "User is consulting a lawyer for a property dispute", "metadata": { "category": "Property", "case_status": "Active" }, "event": "ADD" }
        ]
    }

**Example 2 - Updating Metadata**
- Old Memory:
    [
        { "id": "0", "text": "User filed a divorce petition", "metadata": { "category": "Family", "case_status": "Pending" } }
    ]
- Retrieved facts: ["User's divorce petition was accepted by the court"]
- Retrieved metadata: { "category": "Family", "case_status": "Active" }
- New Memory:
    {
        "memory": [
            {
                "id": "0",
                "text": "User's divorce petition was accepted by the court",
                "metadata": { "category": "Family", "case_status": "Active" },
                "event": "UPDATE",
                "old_memory": "User filed a divorce petition"
            }
        ]
    }

**Example 3 - Deleting Metadata**
- Old Memory:
    [
        { "id": "0", "text": "User is pursuing a wrongful termination lawsuit", "metadata": { "category": "Civil", "case_status": "Active" } }
    ]
- Retrieved facts: ["User's wrongful termination case was dismissed"]
- Retrieved metadata: { "category": "Civil", "case_status": "Closed" }
- New Memory:
    {
        "memory": [
            { "id": "0", "text": "User is pursuing a wrongful termination lawsuit", "metadata": { "category": "Civil", "case_status": "Active" }, "event": "DELETE" }
        ]
    }

---

## Guidelines
1. Always include both `text` and `metadata` in memory entries.
2. Metadata should be updated along with the text when case category or status changes.
3. If case status progresses (e.g., Pending → Active → Closed), treat it as an **UPDATE**.
4. If metadata contradicts old data (e.g., "Closed" vs "Active" simultaneously), mark the old entry as **DELETE** and add/update accordingly.
5. Use consistent JSON formatting that can be parsed directly by the system.

Return output strictly in this JSON format:
{
  "memory": [
    { "id": "...", "text": "...", "metadata": { "category": "...", "case_status": "..." }, "event": "ADD|UPDATE|DELETE|NONE", "old_memory": "..." (if update) }
  ]
}
"""

#Aceesing the Api keys of Pinecone and Gemini Ai
Pinecone_api_key = os.getenv("Pinecone_api_key")
#initilizing the configurations
config = {
    "llm": {
        "provider": "gemini",
        "config": {
            "model": "gemini-2.0-flash-001",
            "temperature": 0.2,
            "max_tokens": 2000,
            "top_p": 1.0
        }
    },
    "embedder": {
        "provider": "gemini",
        "config": {
            "model": "models/text-embedding-004",
        }
},
"vector_store": {
        "provider": "pinecone",
        "config": {
            "collection_name": "legalchatbot",
            "embedding_model_dims": 768,  
            "namespace": "my-namespace", 
            "api_key": Pinecone_api_key,
            "serverless_config": {
                "cloud": "aws", 
                "region": "us-east-1"
            },
            "metric": "cosine"
        }
},
"custom_update_prompt": UPDATE_MEMORY_PROMPT,
"version": "v1.1"

}
#Initilizing the Memory using above Configuration
m=Memory.from_config(config)