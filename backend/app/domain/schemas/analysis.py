from pydantic import BaseModel, Field
from typing import Literal


class DocumentMetadata(BaseModel):
    """
    Structured output model for document analysis via instructor.
    Instructor enforces schema compliance so the LLM can never return
    a malformed dict — every field is validated at extraction time.
    """
    key_points: list[str] = Field(
        ...,
        min_length=2,
        max_length=6,
        description="2-6 bullet-point key takeaways from the document",
    )
    document_type: str = Field(
        ...,
        description="Type of document: report, contract, research paper, financial statement, invoice, etc.",
    )
    contains_financial_data: bool = Field(
        ...,
        description="True if the document contains financial figures, revenue, costs, budgets, etc.",
    )
    contains_personal_data: bool = Field(
        ...,
        description="True if the document contains PII such as names, emails, phone numbers, addresses.",
    )
    language: str = Field(default="English", description="Primary language of the document")
    estimated_reading_time_minutes: int = Field(
        ...,
        ge=1,
        description="Estimated reading time in minutes at 200 words per minute",
    )


class ChunkingComparison(BaseModel):
    """Records the outcome of comparing fixed-size vs semantic chunking strategies."""
    fixed_chunk_count: int
    semantic_chunk_count: int
    avg_fixed_chars: int
    avg_semantic_chars: int
    recommended_strategy: Literal["fixed", "semantic"]
    reason: str
