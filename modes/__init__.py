from modes.base import MemoryMode
from modes.full_context import FullContextMode
from modes.ours import OursMode, RunJob
from modes.rag import RagMode

MODE_NAMES = ("full", "rag", "ours")


def get_mode(name: str, run_job: RunJob | None = None) -> MemoryMode:
    if name == "full":
        return FullContextMode()
    if name == "rag":
        return RagMode()
    if name == "ours":
        return OursMode(run_job)
    raise ValueError(f"unknown mode {name!r}; choose from {MODE_NAMES}")
