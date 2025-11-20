from typing import Any, Optional


def _safe_int(value: Any) -> Optional[int]:
    return int(value) if isinstance(value, (int, float)) else None


def func(persisted: Optional[dict[str, Any]]) -> None:
    if not persisted:
        return

    val = persisted.get("key")
    _safe_int(val)
