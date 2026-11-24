from __future__ import annotations

import os


def restricted_cpu_threads() -> int | None:
    """Use explicit ORT threads when Linux restricts the process CPU affinity."""
    if not hasattr(os, "sched_getaffinity"):
        return None
    available = len(os.sched_getaffinity(0))
    return available if available < (os.cpu_count() or available) else None
