from __future__ import annotations

import time
from typing import Callable

from serving.contracts import ErrorCode


class CircuitBreaker:
    def __init__(self, clock: Callable[[], float] | None = None, threshold: int = 5, cooldown_s: float = 30.0) -> None:
        self.clock = clock or time.monotonic
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self.failures = 0
        self.opened_at: float | None = None
        self.probe_used = False

    def allow(self) -> bool:
        if self.opened_at is None:
            return True
        if self.clock() - self.opened_at < self.cooldown_s:
            return False
        if self.probe_used:
            return False
        self.probe_used = True
        return True

    def record(self, error_code: ErrorCode | None) -> None:
        if error_code in {ErrorCode.INTERNAL, ErrorCode.RESOURCE_EXHAUSTED}:
            self.failures += 1
            if self.failures >= self.threshold:
                self.opened_at = self.clock()
                self.probe_used = False
            return
        if error_code is None:
            self.failures = 0
            self.opened_at = None
            self.probe_used = False
