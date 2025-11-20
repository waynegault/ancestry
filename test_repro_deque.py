from collections import deque
from typing import Any, Dict, cast


class A:
    def __init__(self) -> None:
        self.q: deque[dict[str, Any]] = deque()
        self.q2 = cast(deque[dict[str, Any]], deque())

    def add(self) -> None:
        self.q.append({"a": 1})
        self.q2.append({"a": 1})
