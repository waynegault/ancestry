#!/usr/bin/env python3
"""Lightweight database schema migration runner."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from standard_imports import setup_module

logger = setup_module(globals(), __name__)

MigrationFn = Callable[[Engine], None]


@dataclass(frozen=True)
class Migration:
    """Represents a single schema migration."""

    version: str
    description: str
    upgrade: MigrationFn
    downgrade: Optional[MigrationFn] = None


class MigrationRegistry:
    """In-memory registry that keeps migrations ordered by version string."""

    def __init__(self) -> None:
        self._migrations: list[Migration] = []

    def register(self, migration: Migration) -> None:
        if any(existing.version == migration.version for existing in self._migrations):
            raise ValueError(f"Duplicate migration version: {migration.version}")
        self._migrations.append(migration)
        self._migrations.sort(key=lambda m: m.version)
        logger.debug("Registered migration %s - %s", migration.version, migration.description)

    def migrations(self) -> list[Migration]:
        return list(self._migrations)

    def clear(self) -> None:
        self._migrations.clear()


_REGISTRY = MigrationRegistry()


def register_migration(migration: Migration) -> None:
    """Register a migration with the default registry."""

    _REGISTRY.register(migration)


def get_registered_migrations() -> list[Migration]:
    """Return a copy of registered migrations (for diagnostics/tests)."""

    return _REGISTRY.migrations()


def _ensure_version_table(engine: Engine) -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version TEXT PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )
                """
            )
        )


def _fetch_applied_versions(engine: Engine) -> set[str]:
    _ensure_version_table(engine)
    with engine.begin() as connection:
        rows = connection.execute(text("SELECT version FROM schema_migrations"))
        return {row[0] for row in rows}


def _record_applied_version(engine: Engine, version: str) -> None:
    with engine.begin() as connection:
        connection.execute(
            text("INSERT INTO schema_migrations (version, applied_at) VALUES (:version, :applied_at)"),
            {
                "version": version,
                "applied_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
        )


def apply_pending_migrations(engine: Engine, *, registry: MigrationRegistry | None = None) -> list[str]:
    """Apply all pending migrations for the given engine and return applied versions."""

    active_registry = registry or _REGISTRY
    pending_versions: list[str] = []

    try:
        applied_versions = _fetch_applied_versions(engine)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Unable to read schema_migrations table: %s", exc, exc_info=True)
        return pending_versions

    for migration in active_registry.migrations():
        if migration.version in applied_versions:
            continue
        logger.info("Applying migration %s - %s", migration.version, migration.description)
        migration.upgrade(engine)
        _record_applied_version(engine, migration.version)
        pending_versions.append(migration.version)

    if pending_versions:
        logger.info("Applied %d schema migration(s)", len(pending_versions))
    return pending_versions


def get_applied_versions(engine: Engine) -> list[str]:
    """Return sorted list of applied migration versions."""

    return sorted(_fetch_applied_versions(engine))


def _baseline_upgrade(engine: Engine) -> None:
    """Baseline migration that simply records the current schema."""

    with engine.begin() as connection:
        connection.execute(text("SELECT 1"))


register_migration(
    Migration(
        version="0001_baseline",
        description="Record baseline schema prior to versioning",
        upgrade=_baseline_upgrade,
    )
)


# === Module Tests ===


def _build_temp_engine() -> Engine:
    from sqlalchemy import create_engine
    from sqlalchemy.pool import StaticPool

    return create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


def _test_registry_orders_migrations() -> None:
    registry = MigrationRegistry()

    def _noop(engine: Engine) -> None:  # pragma: no cover - used in tests
        with engine.begin() as connection:
            connection.execute(text("SELECT 1"))

    registry.register(Migration("0002", "second", _noop))
    registry.register(Migration("0001", "first", _noop))

    versions = [migration.version for migration in registry.migrations()]
    assert versions == ["0001", "0002"], versions


def _test_apply_pending_migrations_executes_once() -> None:
    registry = MigrationRegistry()

    def _create_table(engine: Engine) -> None:
        with engine.begin() as connection:
            connection.execute(text("CREATE TABLE IF NOT EXISTS demo (id INTEGER PRIMARY KEY)"))

    registry.register(Migration("1000_demo", "Create demo table", _create_table))

    engine = _build_temp_engine()
    applied_first = apply_pending_migrations(engine, registry=registry)
    assert applied_first == ["1000_demo"]
    applied_second = apply_pending_migrations(engine, registry=registry)
    assert applied_second == []
    versions = get_applied_versions(engine)
    assert versions == ["1000_demo"], versions


def schema_migrator_module_tests() -> bool:
    from test_framework import TestSuite, suppress_logging

    with suppress_logging():
        suite = TestSuite("Schema Migrator", __name__)
        suite.start_suite()
        suite.run_test(
            "Registry enforces ordering",
            _test_registry_orders_migrations,
            "Migrations should be sorted lexicographically",
            "ordering",
            "Register out-of-order migrations and ensure the list is sorted",
        )
        suite.run_test(
            "Apply pending migrations once",
            _test_apply_pending_migrations_executes_once,
            "Migration should run strictly once and record version",
            "apply",
            "Run apply twice and confirm only first call modifies schema",
        )
        return suite.finish_suite()


from test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(schema_migrator_module_tests)


if __name__ == "__main__":
    if schema_migrator_module_tests():
        sys.exit(0)
    sys.exit(1)
