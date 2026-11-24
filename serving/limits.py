from dataclasses import dataclass


@dataclass
class Limits:
    max_edge: int = 1920
    queue_capacity: int = 4
    max_in_flight: int = 2
    timeout_s: float = 2.0
