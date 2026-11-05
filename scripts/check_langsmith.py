"""Diagnose LangSmith auth without printing the key.

Tries LANGSMITH_ENDPOINT (if set) plus the known regional endpoints.
Run:  docker compose run --rm api python -m scripts.check_langsmith
"""

import os
import socket
from urllib.parse import urlparse

import requests

raw = os.environ.get("LANGSMITH_API_KEY", "")
key = raw.strip().strip('"').strip("'")
configured = os.environ.get("LANGSMITH_ENDPOINT", "").rstrip("/")
print(f"key length={len(key)} prefix={key[:7]} needed_cleanup={key != raw}")
print(f"LANGSMITH_ENDPOINT={configured or '(not set, defaults to US)'}")

candidates = [configured] if configured else []
candidates += [
    "https://api.smith.langchain.com",
    "https://eu.api.smith.langchain.com",
    "https://apac.api.smith.langchain.com",
    "https://ap.api.smith.langchain.com",
    "https://au.api.smith.langchain.com",
]

for base in dict.fromkeys(candidates):
    host = urlparse(base).hostname
    try:
        socket.gethostbyname(host)
    except OSError:
        print(f"{base} -> host does not exist")
        continue
    try:
        r = requests.get(f"{base}/api/v1/sessions", params={"limit": 1}, headers={"x-api-key": key}, timeout=15)
        print(f"{base} -> {r.status_code} {r.text[:80]}")
    except requests.RequestException as e:
        print(f"{base} -> error {e.__class__.__name__}")
