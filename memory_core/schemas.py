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


class EntityAlias(BaseModel):
    name: str   # canonical name, e.g. "Rahul"
    alias: str  # how the user also refers to it, e.g. "my cofounder"


class ExtractedNotes(BaseModel):
    user_name: str | None = None  # the user's own name, if stated
    notes: list[ExtractedNote] = []
    aliases: list[EntityAlias] = []
    forget: list[str] = []       # descriptions of things the user asked to forget (FR-S15)
    corrections: list[str] = []  # descriptions of earlier statements the user said were wrong (FR-S15)


class SubQuery(BaseModel):
    search_query: str
    entities: list[str] = []
    time_from: date | None = None
    time_to: date | None = None
    needs_history: bool = False


class QueryAnalysis(BaseModel):
    # An empty list is accepted (the caller falls back to the raw message) instead of costing a retry.
    queries: list[SubQuery] = Field(default_factory=list, max_length=3)


class ReconcileDecision(BaseModel):
    new: int
    replaces: list[str] = []


class ReconcileResult(BaseModel):
    decisions: list[ReconcileDecision] = []
