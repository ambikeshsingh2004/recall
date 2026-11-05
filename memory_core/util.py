from datetime import datetime


def estimate_tokens(text: str) -> int:
    """Rough token count (about 4 characters per token); good enough for budgets and reports."""
    return len(text) // 4


def fmt_time(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%d %H:%M")


def fmt_date(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%d")
