#!/usr/bin/env python3
"""Retention policies for the on-disk Cache/ hierarchy."""

from __future__ import annotations

# === PATH SETUP FOR PACKAGE IMPORTS ===
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Optional

from core.registry_utils import auto_register_module

logger = logging.getLogger(__name__)
auto_register_module(globals(), __name__)


@dataclass(frozen=True)
class RetentionPolicy:
    """Constraints applied to a cache directory."""

    max_age_hours: Optional[float] = None
    max_total_size_mb: Optional[float] = None
    max_file_count: Optional[int] = None
    file_extensions: tuple[str, ...] = ()
    description: str = ""


@dataclass(frozen=True)
class RetentionTarget:
    """Directory plus retention policy metadata."""

    name: str
    path: Path
    policy: RetentionPolicy
    recursive: bool = True


@dataclass
class FileInfo:
    """File metadata used during retention evaluation."""

    path: Path
    size: int
    mtime: float


@dataclass
class RetentionResult:
    """Outcome of applying a retention policy."""

    name: str
    path: str
    files_scanned: int
    files_remaining: int
    files_deleted: int
    bytes_deleted: int
    total_size_bytes: int
    oldest_file_age_hours: float
    newest_file_age_hours: float
    run_timestamp: float
    duration_ms: float
    auto_triggered: bool
    violations: dict[str, bool]
    policy: RetentionPolicy
    last_error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "name": self.name,
            "path": self.path,
            "files_scanned": self.files_scanned,
            "files_remaining": self.files_remaining,
            "files_deleted": self.files_deleted,
            "bytes_deleted": self.bytes_deleted,
            "total_size_bytes": self.total_size_bytes,
            "oldest_file_age_hours": round(self.oldest_file_age_hours, 2),
            "newest_file_age_hours": round(self.newest_file_age_hours, 2),
            "run_timestamp": self.run_timestamp,
            "duration_ms": round(self.duration_ms, 3),
            "auto_triggered": self.auto_triggered,
            "violations": self.violations,
            "policy": asdict(self.policy),
        }
        if self.last_error:
            data["last_error"] = self.last_error
        return data


@dataclass
class _RetentionRunState:
    """Mutable bookkeeping while enforcing a retention policy."""

    total_size_bytes: int
    files_deleted: int = 0
    bytes_deleted: int = 0


class CacheRetentionService:
    """Applies retention constraints to Cache/ subdirectories."""

    def __init__(self, auto_interval_seconds: int = 3600, register_defaults: bool = True) -> None:
        self._targets: dict[str, RetentionTarget] = {}
        self._last_results: dict[str, RetentionResult] = {}
        self._total_deleted_files = 0
        self._total_deleted_bytes = 0
        self._last_run_ts = 0.0
        self._last_auto_run_ts = 0.0
        self._auto_interval_seconds = auto_interval_seconds
        if register_defaults:
            self._register_default_targets()

    # ---- target management -------------------------------------------------

    def register_target(self, target: RetentionTarget) -> None:
        self._targets[target.name] = target

    # ---- public API -------------------------------------------------------

    def apply_policies(
        self, target_name: Optional[str] = None, *, auto_triggered: bool = False
    ) -> dict[str, RetentionResult]:
        """Apply retention policies and return per-target results."""

        if target_name:
            target = self._targets.get(target_name)
            if not target:
                logger.debug("Retention target %s not found", target_name)
                return {}
            selected = {target_name: target}
        else:
            selected = self._targets
        results: dict[str, RetentionResult] = {}
        for name, target in selected.items():
            result = self._apply_retention(target, auto_triggered=auto_triggered)
            results[name] = result
            self._last_results[name] = result
            self._total_deleted_files += result.files_deleted
            self._total_deleted_bytes += result.bytes_deleted
        if results:
            now = time.time()
            self._last_run_ts = now
            if auto_triggered:
                self._last_auto_run_ts = now
        return results

    def get_summary(self) -> dict[str, Any]:
        """Return retention statistics suitable for dashboards."""

        self._ensure_recent_auto_run()
        targets_summary: list[dict[str, Any]] = []
        for name, target in self._targets.items():
            result = self._last_results.get(name)
            if result is None:
                result = self._probe_without_cleanup(target)
                self._last_results[name] = result
            targets_summary.append(result.to_dict())

        return {
            "targets": targets_summary,
            "last_run_ts": self._last_run_ts,
            "auto_interval_minutes": round(self._auto_interval_seconds / 60, 1),
            "total_deleted_files": self._total_deleted_files,
            "total_deleted_mb": round(self._total_deleted_bytes / (1024 * 1024), 3),
        }

    # ---- internal helpers -------------------------------------------------

    def _ensure_recent_auto_run(self) -> None:
        now = time.time()
        if now - self._last_auto_run_ts >= self._auto_interval_seconds:
            self.apply_policies(auto_triggered=True)

    def _probe_without_cleanup(self, target: RetentionTarget) -> RetentionResult:
        files = self._scan_files(target)
        total_size = sum(f.size for f in files)
        return self._build_result(
            target=target,
            files_scanned=len(files),
            files_remaining=len(files),
            files_deleted=0,
            bytes_deleted=0,
            total_size_bytes=total_size,
            auto_triggered=False,
            violations={"age": False, "size": False, "count": False},
            start_time=time.time(),
            remaining_files=files,
        )

    @staticmethod
    def _scan_files(target: RetentionTarget) -> list[FileInfo]:
        files: list[FileInfo] = []
        search_path = target.path
        if not search_path.exists():
            return files

        iterator = search_path.rglob("*") if target.recursive else search_path.glob("*")
        extensions = {
            ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in target.policy.file_extensions
        }

        for entry in iterator:
            try:
                if not entry.is_file():
                    continue
                if extensions and entry.suffix.lower() not in extensions:
                    continue
                stat_info = entry.stat()
                files.append(FileInfo(path=entry, size=stat_info.st_size, mtime=stat_info.st_mtime))
            except FileNotFoundError:
                continue
            except PermissionError as perm_err:
                logger.debug("Permission denied scanning %s: %s", entry, perm_err)
        return files

    def _apply_retention(self, target: RetentionTarget, *, auto_triggered: bool) -> RetentionResult:
        start_time = time.time()
        files = self._scan_files(target)
        initial_count = len(files)
        state = _RetentionRunState(total_size_bytes=sum(f.size for f in files))
        violations = {"age": False, "size": False, "count": False}

        files = self._apply_age_policy(files, target.policy, state, violations)
        files = self._apply_size_policy(files, target.policy, state, violations)
        files = self._apply_count_policy(files, target.policy, state, violations)

        return self._build_result(
            target=target,
            files_scanned=initial_count,
            files_remaining=len(files),
            files_deleted=state.files_deleted,
            bytes_deleted=state.bytes_deleted,
            total_size_bytes=state.total_size_bytes,
            auto_triggered=auto_triggered,
            violations=violations,
            start_time=start_time,
            remaining_files=files,
        )

    @staticmethod
    def _delete_file(info: FileInfo) -> tuple[bool, int]:
        try:
            info.path.unlink()
            logger.debug("Retention removed %s", info.path)
            return True, info.size
        except FileNotFoundError:
            return True, 0
        except Exception as exc:
            logger.debug("Failed to delete %s: %s", info.path, exc)
            return False, 0

    def _apply_age_policy(
        self,
        files: list[FileInfo],
        policy: RetentionPolicy,
        state: _RetentionRunState,
        violations: dict[str, bool],
    ) -> list[FileInfo]:
        if policy.max_age_hours is None:
            return files

        cutoff = time.time() - (policy.max_age_hours * 3600)
        remaining: list[FileInfo] = []
        for info in files:
            if info.mtime >= cutoff:
                remaining.append(info)
                continue

            success, removed_bytes = self._delete_file(info)
            if success:
                state.files_deleted += 1
                state.bytes_deleted += removed_bytes
                state.total_size_bytes -= info.size
                violations["age"] = True
            else:
                remaining.append(info)
        return remaining

    def _apply_size_policy(
        self,
        files: list[FileInfo],
        policy: RetentionPolicy,
        state: _RetentionRunState,
        violations: dict[str, bool],
    ) -> list[FileInfo]:
        max_mb = policy.max_total_size_mb
        if max_mb is None:
            return files

        max_bytes = int(max_mb * 1024 * 1024)
        if state.total_size_bytes <= max_bytes:
            return files

        violations["size"] = True
        files = sorted(files, key=lambda f: f.mtime)
        idx = 0
        while state.total_size_bytes > max_bytes and idx < len(files):
            info = files[idx]
            success, removed_bytes = self._delete_file(info)
            if success:
                state.files_deleted += 1
                state.bytes_deleted += removed_bytes
                state.total_size_bytes -= info.size
                files.pop(idx)
            else:
                idx += 1
        return files

    def _apply_count_policy(
        self,
        files: list[FileInfo],
        policy: RetentionPolicy,
        state: _RetentionRunState,
        violations: dict[str, bool],
    ) -> list[FileInfo]:
        max_count = policy.max_file_count
        if max_count is None or len(files) <= max_count:
            return files

        violations["count"] = True
        files = sorted(files, key=lambda f: f.mtime)
        failures = 0
        while len(files) > max_count and files:
            info = files[0]
            success, removed_bytes = self._delete_file(info)
            if success:
                state.files_deleted += 1
                state.bytes_deleted += removed_bytes
                state.total_size_bytes -= info.size
                files.pop(0)
                failures = 0
            else:
                files.append(files.pop(0))
                failures += 1
                if failures >= len(files):
                    break
        return files

    def _build_result(
        self,
        *,
        target: RetentionTarget,
        files_scanned: int,
        files_remaining: int,
        files_deleted: int,
        bytes_deleted: int,
        total_size_bytes: int,
        auto_triggered: bool,
        violations: dict[str, bool],
        start_time: float,
        remaining_files: Optional[list[FileInfo]] = None,
        last_error: Optional[str] = None,
    ) -> RetentionResult:
        now = time.time()
        oldest_age_hours = 0.0
        newest_age_hours = 0.0
        infos = remaining_files
        if infos is None and files_remaining > 0:
            infos = self._scan_files(target)

        if infos:
            oldest_mtime = min(f.mtime for f in infos)
            newest_mtime = max(f.mtime for f in infos)
            oldest_age_hours = max(0.0, (now - oldest_mtime) / 3600)
            newest_age_hours = max(0.0, (now - newest_mtime) / 3600)
            remaining_count = len(infos)
        else:
            remaining_count = files_remaining
        return RetentionResult(
            name=target.name,
            path=str(target.path),
            files_scanned=files_scanned,
            files_remaining=remaining_count,
            files_deleted=files_deleted,
            bytes_deleted=bytes_deleted,
            total_size_bytes=total_size_bytes,
            oldest_file_age_hours=oldest_age_hours,
            newest_file_age_hours=newest_age_hours,
            run_timestamp=now,
            duration_ms=(now - start_time) * 1000,
            auto_triggered=auto_triggered,
            violations=violations,
            policy=target.policy,
            last_error=last_error,
        )

    def _register_default_targets(self) -> None:
        cache_root = Path("Cache")
        defaults = [
            RetentionTarget(
                name="performance_cache",
                path=cache_root / "performance",
                policy=RetentionPolicy(
                    max_age_hours=72,
                    max_total_size_mb=512,
                    file_extensions=(".pkl",),
                    description="GEDCOM performance cache pickles",
                ),
            ),
            RetentionTarget(
                name="session_checkpoints",
                path=cache_root / "session_checkpoints",
                policy=RetentionPolicy(
                    max_age_hours=24 * 7,
                    max_file_count=25,
                    file_extensions=(".json",),
                    description="Session checkpoint archives",
                ),
            ),
            RetentionTarget(
                name="session_state",
                path=cache_root / "session_state",
                policy=RetentionPolicy(
                    max_age_hours=36,
                    max_file_count=5,
                    file_extensions=(".json",),
                    description="Crash-recovery session state",
                ),
                recursive=False,
            ),
        ]
        for target in defaults:
            self.register_target(target)


class ServiceState:
    _service: Optional[CacheRetentionService] = None


def get_retention_service() -> CacheRetentionService:
    if ServiceState._service is None:
        ServiceState._service = CacheRetentionService()
    return ServiceState._service


def enforce_retention_now(target: Optional[str] = None) -> dict[str, dict[str, Any]]:
    service = get_retention_service()
    results = service.apply_policies(target_name=target)
    return {name: result.to_dict() for name, result in results.items()}


def auto_enforce_retention(target: Optional[str] = None) -> None:
    """Run retention with auto-trigger semantics (no return value)."""

    service = get_retention_service()
    service.apply_policies(target_name=target, auto_triggered=True)


def get_retention_summary() -> dict[str, Any]:
    service = get_retention_service()
    return service.get_summary()


# === Module Tests ===


def _touch_file(path: Path, size: int, age_hours: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(os.urandom(size))
    timestamp = time.time() - (age_hours * 3600)
    os.utime(path, (timestamp, timestamp))


def _test_age_based_cleanup() -> None:
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmp:
        service = CacheRetentionService(register_defaults=False)
        target = RetentionTarget(
            name="tmp",
            path=Path(tmp),
            policy=RetentionPolicy(max_age_hours=1, file_extensions=(".json",)),
        )
        service.register_target(target)
        _touch_file(Path(tmp) / "old.json", 32, age_hours=2)
        _touch_file(Path(tmp) / "new.json", 32, age_hours=0.1)
        results = service.apply_policies()
        result = results["tmp"]
        assert result.files_deleted == 1
        assert result.files_remaining == 1


def _test_size_and_count_cleanup() -> None:
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmp:
        service = CacheRetentionService(register_defaults=False)
        target = RetentionTarget(
            name="tmp",
            path=Path(tmp),
            policy=RetentionPolicy(max_total_size_mb=0.001, max_file_count=2, file_extensions=(".bin",)),
        )
        service.register_target(target)
        _touch_file(Path(tmp) / "a.bin", 2048, age_hours=2)
        _touch_file(Path(tmp) / "b.bin", 2048, age_hours=1)
        _touch_file(Path(tmp) / "c.bin", 1024, age_hours=0.5)
        results = service.apply_policies()
        result = results["tmp"]
        assert result.files_remaining <= 2
        assert result.files_deleted >= 1
        assert result.total_size_bytes <= int(0.001 * 1024 * 1024)


def _test_summary_reports_targets() -> None:
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmp:
        service = CacheRetentionService(register_defaults=False)
        target = RetentionTarget(
            name="tmp",
            path=Path(tmp),
            policy=RetentionPolicy(max_age_hours=1),
        )
        service.register_target(target)
        summary = service.get_summary()
        assert summary["targets"], "Summary should include registered target"
        assert summary["targets"][0]["name"] == "tmp"


def cache_retention_module_tests() -> bool:
    from testing.test_framework import TestSuite, suppress_logging

    with suppress_logging():
        suite = TestSuite("Cache Retention", __name__)
        suite.start_suite()
        suite.run_test(
            "Age-based cleanup removes stale files",
            _test_age_based_cleanup,
            "Old files should be deleted when exceeding the max age",
            "age_cleanup",
            "Create files with different ages and enforce policy",
        )
        suite.run_test(
            "Size/count cleanup enforces limits",
            _test_size_and_count_cleanup,
            "Retention ensures total size and file count stay within limits",
            "size_count",
            "Create files exceeding size/count caps and enforce policy",
        )
        suite.run_test(
            "Summary reports registered targets",
            _test_summary_reports_targets,
            "Registry summary should expose configured targets",
            "summary",
            "Create service and verify summary output",
        )
        return suite.finish_suite()


from testing.test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(cache_retention_module_tests)


if __name__ == "__main__":
    raise SystemExit(0 if cache_retention_module_tests() else 1)
