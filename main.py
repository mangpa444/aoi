from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
import os

app = FastAPI()

# Get the Groq API key from environment variable
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set")

client = Groq(api_key=groq_api_key)

class AIRequest(BaseModel):
    model: str
    prompt: str

@app.post("/ai")
async def generate_ai_response(request: AIRequest):
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
        return {"response": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Groq API server is running"}
