"""Centralized backend configuration registry.

All backend URLs live here so dashboard pages never hardcode API endpoints.
Add new backends by appending to BACKENDS — no UI code changes needed.
"""

from __future__ import annotations

import os
from typing import TypedDict


class BackendEntry(TypedDict):
    url: str
    type: str


BACKENDS: dict[str, BackendEntry] = {
    "Local CPU": {
        "url": os.environ.get("AI_API_URL", "http://127.0.0.1:8000/api/v1"),
        "type": "local",
    },
    "Colab GPU": {
        "url": "https://a84a-136-118-99-101.ngrok-free.app/api/v1",
        "type": "remote",
    },
}

DEFAULT_BACKEND = "Local CPU"
