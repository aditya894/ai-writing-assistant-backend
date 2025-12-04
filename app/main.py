from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .llm_client import improve_text

app = FastAPI(
    title="AI Writing Assistant API",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ImproveRequest(BaseModel):
    text: str
    tone: str | None = None
    language: str | None = "en"


class ImproveResponse(BaseModel):
    improved_text: str


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/improve_text", response_model=ImproveResponse)
def improve_text_endpoint(payload: ImproveRequest):
    improved = improve_text(
        text=payload.text,
        tone=payload.tone or "neutral professional",
        language=payload.language or "en",
    )
    return ImproveResponse(improved_text=improved)
