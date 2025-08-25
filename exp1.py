#importing necessary Libraries
import chainlit as cl
from dotenv import load_dotenv
from google import genai
from google.genai import types
import os
#importing memory configurations from memory_llm.py
from memory_llm import m
from Chat_Rag import index
from History_index import history_index
import uuid
from datetime import datetime
import pytz
import json
import re
#initilizing the env file
ist = pytz.timezone("Asia/Kolkata")
load_dotenv()

#Accessing the Gemini AI model using Api key
Gemini_api_key = os.getenv("Gemini_api_key")
client = genai.Client(api_key=Gemini_api_key)
# Get past memory results
def get_past_history(user_id :str):
    past_history = history_index.search(
    namespace="response_for_legalcases",
    query={
        "inputs": {"text": "default"},
        "top_k": 20,
        "filter": {
            "user_id": {"$eq": user_id}
        }
        },
    )
    previous_session_history=[]
    for match in past_history["result"]["hits"]:
        text_data = match["fields"].get("chunk_text")
        if text_data:  
            previous_session_history.append(text_data)
    return previous_session_history
@cl.on_chat_start
async def on_start():
    res = await cl.AskUserMessage(
        content="Welcome! Please enter your username (Vijay or Vinay):",
        timeout=3000
    ).send()
    user_id = res['output'].strip().lower()
    cl.user_session.set("user_id", user_id)
    cl.user_session.set("history", [])
    previous_chat_history=get_past_history(user_id)
    cl.user_session.set("previous_chat_history",previous_chat_history)
#chatbot implementation
@cl.on_message
async def chat(message:cl.Message):
    #Getting info from the Document
    history = cl.user_session.get("history")
    user_id=cl.user_session.get("user_id")
    previous_chat_history=cl.user_session.get("previous_chat_history")
    formatted_prev_chat = "\n".join(previous_chat_history)
    contents = []
    for h in history:
        contents.append(
            types.Content(
                role=h["role"],
                parts=[types.Part(text=h["content"])]
            )
        )
    history.append({"role": "user", "content": message.content})
    Rag_response = index.search(
    namespace="default",
    query={
        "inputs": {"text":message.content},  # 'text' per field_map
        "top_k": 6
    }
    )
    #combinig the chunks into paragraph to give to genai model
    matches = Rag_response["result"]["hits"]
    Rag_content = " ".join([match["fields"]["chunk_text"] for match in matches])

    #retriving the memory
    search_results=m.search(query=message.content,user_id=user_id)
    previous_memory = ""
    if search_results:
        for r in search_results["results"]:
            previous_memory += f"Memory: {r['memory']}\n"
    #generating the content using genail model
    
    system_prompt =f"""
You are a LegalBot assistant.

## Capabilities:
- Interpret user queries in the context of law.
- Use retrieved legal documents, prior session history, and persistent memory to inform your understanding.
- Cite relevant legal authorities from the reference material when applicable.
- Maintain a neutral, professional tone.
- Avoid speculation or invented citations.

## Inputs:
- Persistent Memory (user context): {previous_memory}

## Response Guidelines:
1. Respond only if the query is legal in nature. If not, politely decline.
2. Do not mention memory or history in your response.
3. Base your analysis strictly on retrieved reference material.
4. Provide clear, structured legal reasoning.
5. Include disclaimers:
   - You are not a lawyer.
   - This is for informational purposes only.
   - Recommend consulting a qualified attorney for official advice.

## Classification Task (for backend use only):
After generating the legal response, silently append a JSON object on a last line of the response with the following format:
{{"category": "...", "case_status": "..."}}

Where:
- category is one of: User_Info, Criminal Law, Contract Law, Family Law, Property Law, Labor Law, Tax Law, Corporate Law, Other
- case_status is either: "ongoing", "completed", "not_applicable"

Do NOT include this classification in the visible response. It will be extracted separately for backend processing.

Now, generate a clear, professional legal response to the latest user query.

## Output Format:
1. Response for the User Query
2. On a new line at the end of the response give the Classification task answer
"""


    response=client.models.generate_content_stream(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(
            system_instruction=system_prompt),
            contents = f"Previous history of the session and query is {contents} and  Reference Material:{Rag_content}."
    )
    full = ""
    streamed_msg = cl.Message(content="")
    await streamed_msg.send()
    for chunk in response:
        if chunk.text:
            full+=chunk.text
            await streamed_msg.stream_token(chunk.text)
    match = re.search(r'\{.*"category".*"case_status".*\}', full)
    if match:
        classification = json.loads(match.group())
        print(classification)
        full = full.replace(match.group(), "").strip()
    else:
        classification = {"category": "Other", "case_status": "Unclear"}
    
    streamed_msg.content = full
    await streamed_msg.update()
    m.add(messages=[{
        "role": "user",
        "content":message.content}],
        user_id=user_id,
        metadata={
            "category":classification['category'],
            "case_status":classification['case_status']
        } 
           
    )
    combined_text = f"User: {message.content}\nAssistant: {full}"
    record = {
    "id": str(uuid.uuid4()),
    "user_id": str(user_id).lower(),
    "user_query": message.content,
    "assistant_response": full,
    "chunk_text": combined_text, 
    "timestamp": datetime.now(ist).isoformat()
}
    history_index.upsert_records(records=[record],namespace="response_for_legalcases")