from datetime import UTC, datetime


def parse_time(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
