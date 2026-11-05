"""Shared interface for the three memory modes: full, rag, ours (PRD §12)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from memory_core import messages
from memory_core.config import get_settings


@dataclass
class Message:
    role: Literal["user", "assistant"]
    content: str


@dataclass
class Context:
    memory: str          # memory section placed in the shared system prompt
    messages: list[dict]  # raw window of the current thread + the new user message
    debug: dict = field(default_factory=dict)


class MemoryMode(ABC):
    name: str

    @abstractmethod
    def ingest(self, user_id: str, thread_id: str, message: Message, now: datetime) -> int: ...

    def end_thread(self, user_id: str, thread_id: str, now: datetime) -> None:  # noqa: B027 - optional hook
        """Called when a thread goes idle or the user switches threads."""

    @abstractmethod
    def get_context(self, user_id: str, thread_id: str, question: str, now: datetime) -> Context: ...


def thread_messages(user_id: str, thread_id: str, question: str) -> list[dict]:
    """Current-thread raw window plus the new question, identical for every mode (FR-M2).

    If the question was already saved as the last message (API flow, FR-S1), it is not repeated.
    """
    window = messages.window(user_id, thread_id, get_settings().raw_window_messages)
    if window and window[-1]["role"] == "user" and window[-1]["content"] == question:
        window = window[:-1]
    return [*messages.as_chat(window), {"role": "user", "content": question}]
