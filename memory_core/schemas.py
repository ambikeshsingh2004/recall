"""Pydantic models for LLM JSON outputs (PRD FR-S3, FR-R2)."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class ExtractedNote(BaseModel):
    type: Literal["fact", "event", "instruction"]
    text: str
    entities: list[str] = []
    importance: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    occurred_at: date | None = None


class ExtractedNotes(BaseModel):
    notes: list[ExtractedNote]


class SubQuery(BaseModel):
    search_query: str
    entities: list[str] = []
    time_from: date | None = None
    time_to: date | None = None
    needs_history: bool = False


class QueryAnalysis(BaseModel):
    queries: list[SubQuery] = Field(min_length=1, max_length=3)


class ReconcileDecision(BaseModel):
    new: int
    replaces: list[str] = []


class ReconcileResult(BaseModel):
    decisions: list[ReconcileDecision] = []
