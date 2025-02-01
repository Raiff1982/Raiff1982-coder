from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
from ai_core import AICore

app = FastAPI()

# Initialize the AI core
ai_core = AICore()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    insights: list
    response: str
    security_level: int
    safety_checks: dict
    health_status: dict
    encrypted_query: str

@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    try:
        response = await ai_core.generate_response(request.query)
        return QueryResponse(**response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("shutdown")
async def shutdown_event():
    await ai_core.shutdown()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860) 