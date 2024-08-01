%%writefile main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
import os
from datetime import datetime, timedelta

app = FastAPI()

# Get the Groq API key from environment variable
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set")

client = Groq(api_key=groq_api_key)

class AIRequest(BaseModel):
    model: str
    prompt: str

# Simple in-memory storage for usage statistics
request_count = 0
last_request_time = None

@app.post("/ai")
async def generate_ai_response(request: AIRequest):
    global request_count, last_request_time
    try:
        response = client.chat.completions.create(
            model=request.model,
            messages=[
                {
                    "role": "user",
                    "content": request.prompt,
                }
            ],
        )
        request_count += 1
        last_request_time = datetime.now()
        return {"response": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Groq API server is running"}

@app.get("/status")
async def status():
    global request_count, last_request_time
    return {
        "total_requests": request_count,
        "last_request_time": last_request_time.isoformat() if last_request_time else None,
        "uptime": str(datetime.now() - app.start_time),
        "version": "1.0"
    }

@app.on_event("startup")
async def startup_event():
    app.start_time = datetime.now()