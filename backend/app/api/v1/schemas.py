from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, EmailStr

# --- AUTHENTICATION ---

class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)
    full_name: str | None = None


class ResgisterResponse(BaseModel):
    id: str
    email: EmailStr
    full_name: str | None = None
    role: str
    access_token: str | None = None
    refresh_token: str | None = None
    token_type: str = "bearer"
    message: str


# --- DOCUMENTS ---

class DocumentUploadResponse(BaseModel):
    message: str
    document_id: str
    task_id: str
    status: str
    file_name: str
    url: str
    owner: EmailStr


class DocumentAnalysisResponse(BaseModel):
    id: str
    file_name: str
    url: str
    owner: EmailStr
    status: str
    analysis_results: dict
    created_at: datetime


# --- RAG ---

class QueryRequest(BaseModel):
    query: str = Field(..., description="The question or query about the document")


class SourceNode(BaseModel):
    text: str
    # BUG FIX: score was required but never returned by query() — now Optional with default
    score: Optional[float] = Field(default=0.0, description="Re-ranker relevance score (0–1)")
    metadata: dict = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceNode]
    # New: CRAG confidence level so callers know when to distrust the answer
    retrieval_confidence: str = Field(
        default="unknown",
        description="Retrieval quality: high | medium | low. Low triggers CRAG fallback prefix.",
    )
