"""Frame processing progress — used by the video pipeline to decide when to persist progress.

Keeping this in the AI layer keeps a single place for "how often we sample progress"
while the database writer stays in the service layer (separation of concerns).
"""

from __future__ import annotations

# Persist progress to DB every N **emitted** frames (after pipeline work for that frame).
DEFAULT_PROGRESS_FRAME_INTERVAL = 10


def should_persist_progress(emitted_frame_count: int, *, every_n: int = DEFAULT_PROGRESS_FRAME_INTERVAL) -> bool:
    """Return True when `emitted_frame_count` (1-based) is a multiple of `every_n`."""
    if emitted_frame_count <= 0:
        return False
    return emitted_frame_count % every_n == 0
