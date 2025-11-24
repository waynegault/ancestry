from __future__ import annotations

import json
import sys
import time
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

if __package__ in {None, ""}:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from config import config_schema
from standard_imports import setup_module
from test_framework import TestSuite, create_standard_test_runner
from test_utilities import atomic_write_file

logger = setup_module(globals(), __name__)

CHECKPOINT_VERSION = "1.0"
_DEFAULT_CHECKPOINT_PATH = Path("Cache/action6_checkpoint.json")


@dataclass(frozen=True)
class GatherCheckpointPlan:
    """Description of how checkpoint resume should operate."""

    enabled: bool
    path: Path
    max_age_hours: int

    def __post_init__(self) -> None:  # pragma: no cover - dataclass wiring
        object.__setattr__(self, "path", Path(self.path))

    def describe(self) -> str:
        """Human readable summary for logging/tests."""

        if not self.enabled:
            return "checkpointing disabled"
        return f"resume checkpoints up to {self.max_age_hours}h old"


class GatherCheckpointService:
    """Checkpoint helper that mirrors the legacy helpers for incremental refactors."""

    def __init__(self, plan: Optional[GatherCheckpointPlan] = None) -> None:
        self._plan = plan or checkpoint_settings()
        self._plans_created = 0

    @property
    def plan(self) -> GatherCheckpointPlan:
        return self._plan

    @property
    def plans_created(self) -> int:
        """Return how many plans have been generated (useful for tests/logging)."""

        return self._plans_created

    def build_plan(self, enabled: bool, max_age_hours: int, path: Optional[Path | str] = None) -> GatherCheckpointPlan:
        if max_age_hours <= 0:
            raise ValueError("max_age_hours must be positive")
        resolved_path = _coerce_checkpoint_path(path) if path is not None else self._plan.path
        plan = GatherCheckpointPlan(enabled=enabled, path=resolved_path, max_age_hours=max_age_hours)
        self._plans_created += 1
        logger.debug(
            "Constructed checkpoint plan",
            extra={
                "enabled": enabled,
                "max_age_hours": max_age_hours,
                "checkpoint_path": str(resolved_path),
                "count": self._plans_created,
            },
        )
        return plan

    def describe_resume_strategy(self, plan: GatherCheckpointPlan) -> str:
        """Return a friendly description for log lines/tests."""

        logger.debug("Describing checkpoint plan", extra={"plans_created": self._plans_created})
        return plan.describe()

    def is_enabled(self) -> bool:
        return checkpoint_enabled(self._plan)

    def clear(self) -> None:
        clear_checkpoint(self._plan)

    def load(self, now: Optional[float] = None) -> Optional[dict[str, Any]]:
        return load_checkpoint(self._plan, now=now)

    def persist(
        self,
        next_page: int,
        last_page_to_process: int,
        total_pages_in_run: int,
        state: MutableMapping[str, Any],
        *,
        now: Optional[float] = None,
    ) -> Optional[dict[str, Any]]:
        return persist_checkpoint(
            next_page=next_page,
            last_page_to_process=last_page_to_process,
            total_pages_in_run=total_pages_in_run,
            state=state,
            plan=self._plan,
            now=now,
        )

    def finalize_after_run(self, state: Mapping[str, Any]) -> None:
        finalize_checkpoint_after_run(state, plan=self._plan)


def _coerce_checkpoint_path(path: Path | str | None) -> Path:
    if path is None:
        return _DEFAULT_CHECKPOINT_PATH
    if isinstance(path, Path):
        return path
    return Path(str(path))


def _ensure_positive_int(value: Any, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def checkpoint_settings() -> GatherCheckpointPlan:
    """Return (enabled, path, max_age_hours) for Action 6 checkpointing."""

    enabled = bool(getattr(config_schema, "enable_action6_checkpointing", True))
    raw_path = getattr(config_schema, "action6_checkpoint_file", _DEFAULT_CHECKPOINT_PATH)
    checkpoint_path = _coerce_checkpoint_path(raw_path)
    max_age_hours = getattr(config_schema, "action6_checkpoint_max_age_hours", 24)
    return GatherCheckpointPlan(
        enabled=enabled,
        path=checkpoint_path,
        max_age_hours=_ensure_positive_int(max_age_hours, 24),
    )


def checkpoint_enabled(plan: Optional[GatherCheckpointPlan] = None) -> bool:
    plan_to_use = plan or checkpoint_settings()
    return plan_to_use.enabled


def clear_checkpoint(plan: Optional[GatherCheckpointPlan] = None) -> None:
    plan_to_use = plan or checkpoint_settings()
    if not plan_to_use.enabled:
        return
    try:
        plan_to_use.path.unlink()
        logger.debug("Cleared Action 6 checkpoint at %s", plan_to_use.path)
    except FileNotFoundError:
        return
    except OSError as exc:  # pragma: no cover - diagnostic only
        logger.debug("Failed to remove Action 6 checkpoint: %s", exc)


def load_checkpoint(
    plan: Optional[GatherCheckpointPlan] = None, *, now: Optional[float] = None
) -> Optional[dict[str, Any]]:
    plan_to_use = plan or checkpoint_settings()
    if not plan_to_use.enabled:
        return None
    checkpoint_path = plan_to_use.path
    if not checkpoint_path.exists():
        return None

    try:
        with checkpoint_path.open(encoding="utf-8") as fh:
            checkpoint_data = json.load(fh)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning("Failed to read Action 6 checkpoint at %s: %s", checkpoint_path, exc)
        return None

    timestamp = checkpoint_data.get("timestamp")
    reference_time = now or time.time()
    if isinstance(timestamp, (int, float)):
        age_hours = max(0.0, (reference_time - float(timestamp)) / 3600)
        if age_hours > plan_to_use.max_age_hours:
            logger.info(
                "Ignoring Action 6 checkpoint because it is %.1f hours old (max %d).",
                age_hours,
                plan_to_use.max_age_hours,
            )
            clear_checkpoint(plan_to_use)
            return None
    else:
        checkpoint_data["timestamp"] = reference_time

    return checkpoint_data


def serialize_checkpoint_snapshot(state: Mapping[str, Any]) -> dict[str, int]:
    """Create a lightweight snapshot of processing counters for checkpoint storage."""

    return {
        "total_new": int(state.get("total_new", 0)),
        "total_updated": int(state.get("total_updated", 0)),
        "total_skipped": int(state.get("total_skipped", 0)),
        "total_errors": int(state.get("total_errors", 0)),
        "total_pages_processed": int(state.get("total_pages_processed", 0)),
    }


def write_checkpoint_state(
    next_page: int,
    last_page_to_process: int,
    total_pages_in_run: int,
    state: Mapping[str, Any],
    *,
    plan: Optional[GatherCheckpointPlan] = None,
    now: Optional[float] = None,
) -> Optional[dict[str, Any]]:
    """Persist checkpoint information for later resume and return the payload."""

    plan_to_use = plan or checkpoint_settings()
    if not plan_to_use.enabled:
        return None

    checkpoint_path = plan_to_use.path
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": CHECKPOINT_VERSION,
        "timestamp": now or time.time(),
        "next_page": max(1, int(next_page)),
        "last_page": max(1, int(last_page_to_process)),
        "total_pages_in_run": max(0, int(total_pages_in_run)),
        "start_page": int(state.get("effective_start_page", 1)),
        "requested_start": state.get("requested_start_page"),
        "counters": serialize_checkpoint_snapshot(state),
    }

    try:
        with atomic_write_file(checkpoint_path, mode="w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        logger.debug(
            "Checkpoint saved. Next page=%s (run length=%s).",
            payload["next_page"],
            payload["total_pages_in_run"],
        )
        return payload
    except Exception as exc:  # pragma: no cover - diagnostics only
        if checkpoint_path.exists():
            logger.debug(
                "Checkpoint persisted despite exception; continuing (error: %s)",
                exc,
            )
        else:
            logger.debug("Failed to persist Action 6 checkpoint: %s", exc)
        return None


def persist_checkpoint(
    *,
    next_page: int,
    last_page_to_process: int,
    total_pages_in_run: int,
    state: MutableMapping[str, Any],
    plan: Optional[GatherCheckpointPlan] = None,
    now: Optional[float] = None,
) -> Optional[dict[str, Any]]:
    """Update checkpoint state after finishing or attempting a page."""

    plan_to_use = plan or checkpoint_settings()
    if not plan_to_use.enabled or total_pages_in_run <= 0:
        return None

    if next_page > last_page_to_process:
        clear_checkpoint(plan_to_use)
        return None

    payload = write_checkpoint_state(
        next_page=next_page,
        last_page_to_process=last_page_to_process,
        total_pages_in_run=total_pages_in_run,
        state=state,
        plan=plan_to_use,
        now=now,
    )
    if payload is not None:
        state["last_checkpoint_written_at"] = now or time.time()
    return payload


def finalize_checkpoint_after_run(state: Mapping[str, Any], plan: Optional[GatherCheckpointPlan] = None) -> None:
    """Clear checkpoint file after a successful run."""

    plan_to_use = plan or checkpoint_settings()
    if not plan_to_use.enabled:
        return
    if state.get("final_success"):
        clear_checkpoint(plan_to_use)


# ---------------------------------------------------------------------------
# Module tests
# ---------------------------------------------------------------------------


def _build_plan(tmp_path: Path, *, enabled: bool = True, max_age_hours: int = 24) -> GatherCheckpointPlan:
    return GatherCheckpointPlan(enabled=enabled, path=tmp_path / "checkpoint.json", max_age_hours=max_age_hours)


def _test_checkpoint_plan_disabled() -> bool:
    service = GatherCheckpointService()
    plan = service.build_plan(enabled=False, max_age_hours=24)
    assert plan.describe() == "checkpointing disabled"
    assert service.describe_resume_strategy(plan) == "checkpointing disabled"
    assert service.plans_created == 1
    return True


def _test_checkpoint_plan_enabled() -> bool:
    service = GatherCheckpointService()
    plan = service.build_plan(enabled=True, max_age_hours=12)
    assert plan.describe() == "resume checkpoints up to 12h old"
    assert service.plans_created == 1
    return True


def _test_persist_and_load_round_trip(tmp_dir: Path) -> bool:
    plan = _build_plan(tmp_dir)
    state: MutableMapping[str, Any] = {
        "effective_start_page": 1,
        "requested_start_page": None,
        "total_new": 5,
        "total_updated": 2,
        "total_skipped": 1,
        "total_errors": 0,
        "total_pages_processed": 3,
    }

    payload = persist_checkpoint(
        next_page=4,
        last_page_to_process=10,
        total_pages_in_run=10,
        state=state,
        plan=plan,
        now=123.0,
    )
    assert payload is not None
    assert plan.path.exists()
    assert abs(state["last_checkpoint_written_at"] - 123.0) < 0.001

    loaded = load_checkpoint(plan, now=124.0)
    assert loaded is not None
    assert loaded["next_page"] == 4
    assert loaded["total_pages_in_run"] == 10
    return True


def _test_expired_checkpoint_clears_file(tmp_dir: Path) -> bool:
    plan = _build_plan(tmp_dir, max_age_hours=1)
    plan.path.parent.mkdir(parents=True, exist_ok=True)
    plan.path.write_text(json.dumps({"timestamp": 0, "next_page": 2}), encoding="utf-8")

    assert load_checkpoint(plan, now=7200.0) is None  # 2 hours later -> expired
    assert not plan.path.exists()
    return True


def _test_clear_checkpoint_handles_missing(tmp_dir: Path) -> bool:
    plan = _build_plan(tmp_dir)
    clear_checkpoint(plan)
    assert not plan.path.exists()
    plan.path.write_text("{}", encoding="utf-8")
    clear_checkpoint(plan)
    assert not plan.path.exists()
    return True


def module_tests() -> bool:
    import tempfile

    suite = TestSuite("actions.gather.checkpoint", "actions/gather/checkpoint.py")
    suite.run_test(
        "Disabled plan",
        _test_checkpoint_plan_disabled,
        "Ensures disabled checkpoint plans describe themselves accurately.",
    )
    suite.run_test(
        "Enabled plan",
        _test_checkpoint_plan_enabled,
        "Ensures enabled checkpoint plans include the age window.",
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        suite.run_test(
            "Persist + load round trip",
            lambda: _test_persist_and_load_round_trip(tmp_path),
            "Ensures checkpoints can be written and read back.",
        )
        suite.run_test(
            "Expired checkpoint clears file",
            lambda: _test_expired_checkpoint_clears_file(tmp_path),
            "Ensures stale checkpoints are ignored and removed.",
        )
        suite.run_test(
            "Clear checkpoint",
            lambda: _test_clear_checkpoint_handles_missing(tmp_path),
            "Ensures clear_checkpoint tolerates missing files.",
        )
    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)


if __name__ == "__main__":
    success = module_tests()
    raise SystemExit(0 if success else 1)
