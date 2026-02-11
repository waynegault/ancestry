#!/usr/bin/env python3
"""Enhanced database schema migration runner with rollback, dry-run, and dependencies."""


import contextlib
import sys
from argparse import ArgumentParser, Namespace
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import logging

logger = logging.getLogger(__name__)

MigrationFn = Callable[[Engine], None]
CliAction = Callable[[Engine, bool], None]  # Second arg is dry_run flag
DEFAULT_DB_PATH = Path("Data/ancestry.db")


@dataclass(frozen=True)
class Migration:
    """Represents a single schema migration with optional rollback and dependencies."""

    version: str
    description: str
    upgrade: MigrationFn
    downgrade: MigrationFn | None = None
    depends_on: tuple[str, ...] = field(default_factory=tuple)

    def has_rollback(self) -> bool:
        """Check if this migration supports rollback."""
        return self.downgrade is not None


class MigrationError(Exception):
    """Base exception for migration errors."""

    pass


class DependencyError(MigrationError):
    """Raised when migration dependencies are not satisfied."""

    pass


class RollbackError(MigrationError):
    """Raised when rollback fails or is not supported."""

    pass


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

    def get(self, version: str) -> Migration | None:
        """Get a migration by version string."""
        for migration in self._migrations:
            if migration.version == version:
                return migration
        return None

    def clear(self) -> None:
        self._migrations.clear()

    def validate_dependencies(self) -> list[str]:
        """Validate all migration dependencies exist. Returns list of errors."""
        errors: list[str] = []
        known_versions = {m.version for m in self._migrations}
        for migration in self._migrations:
            for dep in migration.depends_on:
                if dep not in known_versions:
                    errors.append(f"Migration {migration.version} depends on unknown migration {dep}")
                elif dep >= migration.version:
                    errors.append(f"Migration {migration.version} depends on {dep} which is not earlier")
        return errors


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
                "applied_at": datetime.now(UTC).isoformat(timespec="seconds"),
            },
        )


def _remove_applied_version(engine: Engine, version: str) -> None:
    """Remove a version from the applied migrations table (for rollback)."""
    with engine.begin() as connection:
        connection.execute(
            text("DELETE FROM schema_migrations WHERE version = :version"),
            {"version": version},
        )


def _check_dependencies(migration: Migration, applied_versions: set[str], registry: MigrationRegistry) -> list[str]:
    """Check if all dependencies for a migration are satisfied. Returns list of unmet deps."""
    unmet: list[str] = []
    for dep in migration.depends_on:
        if dep not in applied_versions:
            dep_migration = registry.get(dep)
            if dep_migration is None:
                unmet.append(f"{dep} (not registered)")
            else:
                unmet.append(dep)
    return unmet


def apply_pending_migrations(
    engine: Engine, *, registry: MigrationRegistry | None = None, dry_run: bool = False
) -> list[str]:
    """Apply all pending migrations for the given engine and return applied versions.

    Args:
        engine: SQLAlchemy engine to apply migrations to
        registry: Optional custom registry (uses global registry if None)
        dry_run: If True, only report what would be done without executing

    Returns:
        List of migration versions that were (or would be) applied
    """
    active_registry = registry or _REGISTRY
    pending_versions: list[str] = []

    # Validate dependencies first
    dep_errors = active_registry.validate_dependencies()
    if dep_errors:
        for error in dep_errors:
            logger.error("Dependency error: %s", error)
        raise DependencyError(f"Invalid migration dependencies: {dep_errors}")

    try:
        applied_versions = _fetch_applied_versions(engine)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Unable to read schema_migrations table: %s", exc, exc_info=True)
        return pending_versions

    for migration in active_registry.migrations():
        if migration.version in applied_versions:
            continue

        # Check dependencies
        unmet = _check_dependencies(migration, applied_versions, active_registry)
        if unmet:
            raise DependencyError(f"Migration {migration.version} has unmet dependencies: {unmet}")

        if dry_run:
            logger.info("[DRY-RUN] Would apply migration %s - %s", migration.version, migration.description)
            pending_versions.append(migration.version)
            # Track as "applied" for dependency checking in dry-run
            applied_versions.add(migration.version)
        else:
            logger.info("Applying migration %s - %s", migration.version, migration.description)
            migration.upgrade(engine)
            _record_applied_version(engine, migration.version)
            pending_versions.append(migration.version)
            applied_versions.add(migration.version)

    if pending_versions:
        action = "Would apply" if dry_run else "Applied"
        logger.info("%s %d schema migration(s)", action, len(pending_versions))
    return pending_versions


def rollback_migration(
    engine: Engine, version: str, *, registry: MigrationRegistry | None = None, dry_run: bool = False
) -> bool:
    """Rollback a specific migration version.

    Args:
        engine: SQLAlchemy engine
        version: Migration version to rollback
        registry: Optional custom registry
        dry_run: If True, only report what would be done

    Returns:
        True if rollback succeeded (or would succeed in dry-run)

    Raises:
        RollbackError: If migration doesn't support rollback or isn't applied
    """
    active_registry = registry or _REGISTRY
    migration = active_registry.get(version)

    if migration is None:
        raise RollbackError(f"Migration {version} not found in registry")

    if not migration.has_rollback():
        raise RollbackError(f"Migration {version} does not support rollback (no downgrade function)")

    applied_versions = _fetch_applied_versions(engine)
    if version not in applied_versions:
        raise RollbackError(f"Migration {version} is not currently applied")

    # Check if any applied migration depends on this one
    for other in active_registry.migrations():
        if version in other.depends_on and other.version in applied_versions:
            raise RollbackError(f"Cannot rollback {version}: migration {other.version} depends on it")

    if dry_run:
        logger.info("[DRY-RUN] Would rollback migration %s - %s", migration.version, migration.description)
        return True

    logger.info("Rolling back migration %s - %s", migration.version, migration.description)
    assert migration.downgrade is not None  # Guaranteed by has_rollback() check above
    migration.downgrade(engine)
    _remove_applied_version(engine, version)
    logger.info("Successfully rolled back migration %s", version)
    return True


def rollback_to_version(
    engine: Engine, target_version: str, *, registry: MigrationRegistry | None = None, dry_run: bool = False
) -> list[str]:
    """Rollback all migrations after the target version.

    Args:
        engine: SQLAlchemy engine
        target_version: Version to rollback to (this version will remain applied)
        registry: Optional custom registry
        dry_run: If True, only report what would be done

    Returns:
        List of versions that were (or would be) rolled back
    """
    active_registry = registry or _REGISTRY
    applied_versions = sorted(_fetch_applied_versions(engine), reverse=True)  # Latest first
    rolled_back: list[str] = []

    for version in applied_versions:
        if version <= target_version:
            break  # Stop when we reach target

        migration = active_registry.get(version)
        if migration is None:
            logger.warning("Migration %s is applied but not in registry, skipping", version)
            continue

        if not migration.has_rollback():
            raise RollbackError(f"Cannot rollback to {target_version}: migration {version} has no downgrade function")

        if dry_run:
            logger.info("[DRY-RUN] Would rollback migration %s", version)
            rolled_back.append(version)
        else:
            rollback_migration(engine, version, registry=active_registry, dry_run=False)
            rolled_back.append(version)

    if rolled_back:
        action = "Would rollback" if dry_run else "Rolled back"
        logger.info("%s %d migration(s)", action, len(rolled_back))

    return rolled_back


def get_applied_versions(engine: Engine) -> list[str]:
    """Return sorted list of applied migration versions."""

    return sorted(_fetch_applied_versions(engine))


def _resolve_db_path(path_arg: str | None) -> Path:
    path = Path(path_arg).expanduser() if path_arg else DEFAULT_DB_PATH
    if not path.is_absolute():
        path = Path.cwd() / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _build_cli_engine(path_arg: str | None) -> Engine:
    db_path = _resolve_db_path(path_arg)
    return create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})


def _render_status_table(engine: Engine) -> list[str]:
    applied = set(_fetch_applied_versions(engine))
    lines: list[str] = []
    for migration in get_registered_migrations():
        state = "applied" if migration.version in applied else "pending"
        lines.append(f"{migration.version:<16} {state:<8} {migration.description}")
    return lines


def _list_applied_versions(engine: Engine) -> list[str]:
    return get_applied_versions(engine)


def _print_registered_migrations(engine: Engine, dry_run: bool = False) -> None:
    _ = dry_run  # Not used for list
    print("\nRegistered migrations:")
    for line in _render_status_table(engine):
        print(f"  {line}")


def _print_applied_versions(engine: Engine, dry_run: bool = False) -> None:
    _ = dry_run  # Not used for show
    versions = _list_applied_versions(engine)
    print("\nApplied versions (sorted):")
    if versions:
        for version in versions:
            print(f"  {version}")
    else:
        print("  <none>")


def _apply_pending_migrations_cli(engine: Engine, dry_run: bool = False) -> None:
    applied_versions = apply_pending_migrations(engine, dry_run=dry_run)
    if applied_versions:
        action = "Would apply" if dry_run else "Applied"
        print(f"\n{action} migrations:")
        for version in applied_versions:
            print(f"  {version}")
    else:
        print("\nNo pending migrations; schema already current.")


def _collect_cli_actions(args: Namespace) -> list[CliAction]:
    actions: list[CliAction] = []
    if getattr(args, "list", False):
        actions.append(_print_registered_migrations)
    if getattr(args, "show_applied", False):
        actions.append(_print_applied_versions)
    if getattr(args, "apply", False):
        actions.append(_apply_pending_migrations_cli)
    return actions


def _handle_rollback(engine: Engine, args: Namespace, dry_run: bool) -> int:
    """Handle rollback CLI actions. Returns exit code."""
    target = getattr(args, "rollback_to", None)
    version = getattr(args, "rollback", None)

    if target:
        try:
            rolled_back = rollback_to_version(engine, target, dry_run=dry_run)
            if rolled_back:
                action = "Would rollback" if dry_run else "Rolled back"
                print(f"\n{action} migrations:")
                for ver in rolled_back:
                    print(f"  {ver}")
            else:
                print(f"\nNo migrations to rollback (already at or before {target})")
            return 0
        except RollbackError as e:
            print(f"\nRollback failed: {e}")
            return 1

    if version:
        try:
            rollback_migration(engine, version, dry_run=dry_run)
            action = "Would rollback" if dry_run else "Rolled back"
            print(f"\n{action} migration: {version}")
            return 0
        except RollbackError as e:
            print(f"\nRollback failed: {e}")
            return 1

    return 0


def run_cli(argv: Sequence[str] | None = None) -> int:
    """Enhanced CLI for applying, rolling back, or inspecting schema migrations."""

    parser = ArgumentParser(description="Schema migration utility with rollback support")
    parser.add_argument("--db-path", help="Path to the SQLite database", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--list", action="store_true", help="List migrations and their status")
    parser.add_argument("--apply", action="store_true", help="Apply pending migrations (default action)")
    parser.add_argument("--show-applied", action="store_true", help="Show versions recorded in schema_migrations")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without executing")
    parser.add_argument("--rollback", metavar="VERSION", help="Rollback a specific migration version")
    parser.add_argument("--rollback-to", metavar="VERSION", help="Rollback all migrations after VERSION")
    parser.add_argument("--validate", action="store_true", help="Validate migration dependencies")
    parser.add_argument("--run-tests", action="store_true", help="Run module tests instead of CLI actions")

    args = parser.parse_args(list(argv) if argv is not None else sys.argv[1:])

    if args.run_tests:
        return 0 if schema_migrator_module_tests() else 1

    dry_run = getattr(args, "dry_run", False)

    # Handle validation before other actions
    if getattr(args, "validate", False):
        errors = _REGISTRY.validate_dependencies()
        if errors:
            print("\nDependency validation failed:")
            for error in errors:
                print(f"  ❌ {error}")
            return 1
        print("\n✅ All migration dependencies are valid")
        return 0

    # Handle rollback
    if getattr(args, "rollback", None) or getattr(args, "rollback_to", None):
        engine = _build_cli_engine(args.db_path)
        return _handle_rollback(engine, args, dry_run)

    actions = _collect_cli_actions(args)
    if not actions:
        actions.append(_apply_pending_migrations_cli)

    engine = _build_cli_engine(args.db_path)
    for action in actions:
        action(engine, dry_run)

    return 0


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


def _upgrade_0002(engine: Engine) -> None:
    """Add shared_matches_fetched columns to dna_match."""
    with engine.begin() as connection:
        # Check if columns exist first to avoid errors on re-run
        with contextlib.suppress(Exception):
            connection.execute(text("ALTER TABLE dna_match ADD COLUMN shared_matches_fetched BOOLEAN DEFAULT 0"))
        with contextlib.suppress(Exception):
            connection.execute(text("ALTER TABLE dna_match ADD COLUMN shared_matches_fetched_date DATETIME"))


def _downgrade_0002(engine: Engine) -> None:
    """Remove shared_matches_fetched columns from dna_match."""
    with engine.begin() as connection, contextlib.suppress(Exception):
        connection.execute(text("ALTER TABLE dna_match DROP COLUMN shared_matches_fetched"))
        connection.execute(text("ALTER TABLE dna_match DROP COLUMN shared_matches_fetched_date"))


register_migration(
    Migration(
        version="0002_shared_matches_fetched",
        description="Add tracking columns for shared match fetching",
        upgrade=_upgrade_0002,
        downgrade=_downgrade_0002,
        depends_on=("0001_baseline",),
    )
)


def _upgrade_0003(engine: Engine) -> None:
    """Add tree data columns to dna_match."""
    with engine.begin() as connection:
        with contextlib.suppress(Exception):
            connection.execute(text("ALTER TABLE dna_match ADD COLUMN match_tree_id TEXT"))
        with contextlib.suppress(Exception):
            connection.execute(text("ALTER TABLE dna_match ADD COLUMN match_tree_person_id TEXT"))
        with contextlib.suppress(Exception):
            connection.execute(text("ALTER TABLE dna_match ADD COLUMN has_public_tree BOOLEAN"))
        with contextlib.suppress(Exception):
            connection.execute(text("ALTER TABLE dna_match ADD COLUMN tree_size INTEGER"))


def _downgrade_0003(engine: Engine) -> None:
    """Remove tree data columns from dna_match."""
    with engine.begin() as connection:
        with contextlib.suppress(Exception):
            connection.execute(text("ALTER TABLE dna_match DROP COLUMN match_tree_id"))
        with contextlib.suppress(Exception):
            connection.execute(text("ALTER TABLE dna_match DROP COLUMN match_tree_person_id"))
        with contextlib.suppress(Exception):
            connection.execute(text("ALTER TABLE dna_match DROP COLUMN has_public_tree"))
        with contextlib.suppress(Exception):
            connection.execute(text("ALTER TABLE dna_match DROP COLUMN tree_size"))


register_migration(
    Migration(
        version="0003_add_tree_columns",
        description="Add columns for match tree data (ID, person ID, size, public status)",
        upgrade=_upgrade_0003,
        downgrade=_downgrade_0003,
        depends_on=("0002_shared_matches_fetched",),
    )
)


# --- Migration 0004: Add conversation_state columns for Phase 4 Inbound Engine ---


def _upgrade_0004(engine: Engine) -> None:
    """Add status, safety_flag, and last_intent columns to conversation_state."""
    with engine.begin() as connection:
        # Add status column with default 'ACTIVE'
        with contextlib.suppress(Exception):
            connection.execute(
                text("ALTER TABLE conversation_state ADD COLUMN status VARCHAR NOT NULL DEFAULT 'ACTIVE'")
            )
        # Add safety_flag column with default False (0)
        with contextlib.suppress(Exception):
            connection.execute(text("ALTER TABLE conversation_state ADD COLUMN safety_flag BOOLEAN NOT NULL DEFAULT 0"))
        # Add last_intent column (nullable)
        with contextlib.suppress(Exception):
            connection.execute(text("ALTER TABLE conversation_state ADD COLUMN last_intent VARCHAR"))
        # Create index on status for efficient filtering
        with contextlib.suppress(Exception):
            connection.execute(text("CREATE INDEX ix_conversation_state_status ON conversation_state (status)"))
        # Create index on safety_flag for efficient filtering
        with contextlib.suppress(Exception):
            connection.execute(
                text("CREATE INDEX ix_conversation_state_safety_flag ON conversation_state (safety_flag)")
            )


def _downgrade_0004(_engine: Engine) -> None:
    """Remove Phase 4 columns from conversation_state (SQLite doesn't support DROP COLUMN easily)."""
    # SQLite doesn't support DROP COLUMN in older versions, so we just log a warning
    logger.warning(
        "Downgrade 0004: SQLite doesn't easily support DROP COLUMN. "
        "The columns (status, safety_flag, last_intent) will remain but won't be used."
    )


register_migration(
    Migration(
        version="0004_conversation_state_phase4",
        description="Add status, safety_flag, last_intent columns to conversation_state for Phase 4 Inbound Engine",
        upgrade=_upgrade_0004,
        downgrade=_downgrade_0004,
        depends_on=("0003_add_tree_columns",),
    )
)


# --- Migration 0005: Create missing tables for Phase 4+ features ---


def _upgrade_0005(engine: Engine) -> None:
    """Create missing tables: draft_replies, suggested_facts, data_conflicts, staged_updates, shared_matches."""
    with engine.begin() as connection:
        # Create draft_replies table
        with contextlib.suppress(Exception):
            connection.execute(
                text("""
                CREATE TABLE IF NOT EXISTS draft_replies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    people_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
                    conversation_id VARCHAR NOT NULL,
                    content TEXT NOT NULL,
                    status VARCHAR DEFAULT 'PENDING',
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            )
        with contextlib.suppress(Exception):
            connection.execute(
                text("CREATE INDEX IF NOT EXISTS ix_draft_replies_people_id ON draft_replies(people_id)")
            )
        with contextlib.suppress(Exception):
            connection.execute(
                text("CREATE INDEX IF NOT EXISTS ix_draft_replies_conversation_id ON draft_replies(conversation_id)")
            )
        with contextlib.suppress(Exception):
            connection.execute(text("CREATE INDEX IF NOT EXISTS ix_draft_replies_status ON draft_replies(status)"))

        # Create suggested_facts table
        with contextlib.suppress(Exception):
            connection.execute(
                text("""
                CREATE TABLE IF NOT EXISTS suggested_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    people_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
                    fact_type VARCHAR NOT NULL,
                    original_value TEXT,
                    new_value TEXT NOT NULL,
                    source_message_id VARCHAR,
                    status VARCHAR NOT NULL DEFAULT 'PENDING',
                    confidence_score INTEGER,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            )
        with contextlib.suppress(Exception):
            connection.execute(
                text("CREATE INDEX IF NOT EXISTS ix_suggested_facts_people_id ON suggested_facts(people_id)")
            )
        with contextlib.suppress(Exception):
            connection.execute(
                text("CREATE INDEX IF NOT EXISTS ix_suggested_facts_fact_type ON suggested_facts(fact_type)")
            )
        with contextlib.suppress(Exception):
            connection.execute(text("CREATE INDEX IF NOT EXISTS ix_suggested_facts_status ON suggested_facts(status)"))

        # Create data_conflicts table
        with contextlib.suppress(Exception):
            connection.execute(
                text("""
                CREATE TABLE IF NOT EXISTS data_conflicts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    people_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
                    field_name VARCHAR(100) NOT NULL,
                    existing_value TEXT,
                    new_value TEXT NOT NULL,
                    source VARCHAR(100) NOT NULL DEFAULT 'conversation',
                    source_message_id INTEGER REFERENCES conversation_log(id) ON DELETE SET NULL,
                    confidence_score INTEGER,
                    status VARCHAR NOT NULL DEFAULT 'OPEN',
                    resolution_notes TEXT,
                    resolved_by VARCHAR(100),
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    resolved_at DATETIME
                )
            """)
            )
        with contextlib.suppress(Exception):
            connection.execute(
                text("CREATE INDEX IF NOT EXISTS ix_data_conflicts_people_id ON data_conflicts(people_id)")
            )
        with contextlib.suppress(Exception):
            connection.execute(
                text("CREATE INDEX IF NOT EXISTS ix_data_conflicts_field_name ON data_conflicts(field_name)")
            )
        with contextlib.suppress(Exception):
            connection.execute(text("CREATE INDEX IF NOT EXISTS ix_data_conflicts_status ON data_conflicts(status)"))
        with contextlib.suppress(Exception):
            connection.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_data_conflicts_status_created ON data_conflicts(status, created_at)"
                )
            )
        with contextlib.suppress(Exception):
            connection.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_data_conflicts_person_field ON data_conflicts(people_id, field_name)"
                )
            )

        # Create staged_updates table
        with contextlib.suppress(Exception):
            connection.execute(
                text("""
                CREATE TABLE IF NOT EXISTS staged_updates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    people_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
                    field_name VARCHAR(100) NOT NULL,
                    current_value TEXT,
                    proposed_value TEXT NOT NULL,
                    source VARCHAR(100) NOT NULL DEFAULT 'conversation',
                    source_message_id INTEGER REFERENCES conversation_log(id) ON DELETE SET NULL,
                    confidence_score INTEGER,
                    status VARCHAR NOT NULL DEFAULT 'pending',
                    reviewer_notes TEXT,
                    reviewed_by VARCHAR(100),
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    reviewed_at DATETIME,
                    applied_at DATETIME
                )
            """)
            )
        with contextlib.suppress(Exception):
            connection.execute(
                text("CREATE INDEX IF NOT EXISTS ix_staged_updates_people_id ON staged_updates(people_id)")
            )
        with contextlib.suppress(Exception):
            connection.execute(
                text("CREATE INDEX IF NOT EXISTS ix_staged_updates_field_name ON staged_updates(field_name)")
            )
        with contextlib.suppress(Exception):
            connection.execute(text("CREATE INDEX IF NOT EXISTS ix_staged_updates_status ON staged_updates(status)"))
        with contextlib.suppress(Exception):
            connection.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_staged_updates_status_created ON staged_updates(status, created_at)"
                )
            )
        with contextlib.suppress(Exception):
            connection.execute(
                text("CREATE INDEX IF NOT EXISTS ix_staged_updates_person_status ON staged_updates(people_id, status)")
            )

        # Create shared_matches table
        with contextlib.suppress(Exception):
            connection.execute(
                text("""
                CREATE TABLE IF NOT EXISTS shared_matches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
                    shared_match_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
                    shared_cm INTEGER,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            )
        with contextlib.suppress(Exception):
            connection.execute(
                text("CREATE INDEX IF NOT EXISTS ix_shared_matches_person_id ON shared_matches(person_id)")
            )
        with contextlib.suppress(Exception):
            connection.execute(
                text("CREATE INDEX IF NOT EXISTS ix_shared_matches_shared_match_id ON shared_matches(shared_match_id)")
            )
        with contextlib.suppress(Exception):
            connection.execute(
                text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ix_shared_matches_pair ON shared_matches(person_id, shared_match_id)"
                )
            )


def _downgrade_0005(engine: Engine) -> None:
    """Drop tables created in migration 0005."""
    with engine.begin() as connection:
        with contextlib.suppress(Exception):
            connection.execute(text("DROP TABLE IF EXISTS shared_matches"))
        with contextlib.suppress(Exception):
            connection.execute(text("DROP TABLE IF EXISTS staged_updates"))
        with contextlib.suppress(Exception):
            connection.execute(text("DROP TABLE IF EXISTS data_conflicts"))
        with contextlib.suppress(Exception):
            connection.execute(text("DROP TABLE IF EXISTS suggested_facts"))
        with contextlib.suppress(Exception):
            connection.execute(text("DROP TABLE IF EXISTS draft_replies"))


register_migration(
    Migration(
        version="0005_create_missing_tables",
        description="Create draft_replies, suggested_facts, data_conflicts, staged_updates, shared_matches tables",
        upgrade=_upgrade_0005,
        downgrade=_downgrade_0005,
        depends_on=("0004_conversation_state_phase4",),
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


def _test_dry_run_mode() -> None:
    """Test that dry-run mode doesn't actually apply migrations."""
    registry = MigrationRegistry()

    def _create_table(engine: Engine) -> None:
        with engine.begin() as connection:
            connection.execute(text("CREATE TABLE dry_test (id INTEGER PRIMARY KEY)"))

    registry.register(Migration("2000_dry", "Dry run test", _create_table))

    engine = _build_temp_engine()
    _ensure_version_table(engine)

    # Dry run should report what would be done
    applied = apply_pending_migrations(engine, registry=registry, dry_run=True)
    assert applied == ["2000_dry"], f"Expected ['2000_dry'], got {applied}"

    # But version should NOT be recorded
    versions = get_applied_versions(engine)
    assert "2000_dry" not in versions, f"Dry run should not record version, got {versions}"

    # Table should NOT exist
    with engine.begin() as conn:
        result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='dry_test'"))
        tables = [row[0] for row in result]
    assert "dry_test" not in tables, "Dry run should not create table"


def _test_rollback_single_migration() -> None:
    """Test rolling back a single migration."""
    registry = MigrationRegistry()

    def _create_table(engine: Engine) -> None:
        with engine.begin() as connection:
            connection.execute(text("CREATE TABLE rollback_test (id INTEGER PRIMARY KEY)"))

    def _drop_table(engine: Engine) -> None:
        with engine.begin() as connection:
            connection.execute(text("DROP TABLE IF EXISTS rollback_test"))

    registry.register(Migration("3000_rollback", "Rollback test", _create_table, _drop_table))

    engine = _build_temp_engine()

    # Apply migration
    apply_pending_migrations(engine, registry=registry)
    assert "3000_rollback" in get_applied_versions(engine)

    # Rollback
    result = rollback_migration(engine, "3000_rollback", registry=registry)
    assert result is True
    assert "3000_rollback" not in get_applied_versions(engine)


def _test_rollback_no_downgrade() -> None:
    """Test that rollback fails when no downgrade function exists."""
    registry = MigrationRegistry()

    def _create_table(engine: Engine) -> None:
        with engine.begin() as connection:
            connection.execute(text("CREATE TABLE no_down (id INTEGER PRIMARY KEY)"))

    # No downgrade function
    registry.register(Migration("4000_nodown", "No downgrade", _create_table))

    engine = _build_temp_engine()
    apply_pending_migrations(engine, registry=registry)

    try:
        rollback_migration(engine, "4000_nodown", registry=registry)
        raise AssertionError("Expected RollbackError")
    except RollbackError as e:
        assert "does not support rollback" in str(e)


def _test_dependency_validation() -> None:
    """Test migration dependency validation."""
    registry = MigrationRegistry()

    def _noop(engine: Engine) -> None:
        with engine.begin() as connection:
            connection.execute(text("SELECT 1"))

    # Register migration with invalid dependency
    registry.register(Migration("5000_base", "Base", _noop))
    registry.register(Migration("5001_depends", "Depends on nonexistent", _noop, depends_on=("9999_fake",)))

    errors = registry.validate_dependencies()
    assert len(errors) == 1
    assert "9999_fake" in errors[0]


def _test_dependency_enforcement() -> None:
    """Test that migrations won't apply if dependencies aren't met."""
    registry = MigrationRegistry()

    def _noop(engine: Engine) -> None:
        with engine.begin() as connection:
            connection.execute(text("SELECT 1"))

    # Register migrations with dependency
    registry.register(Migration("6000_base", "Base migration", _noop))
    registry.register(Migration("6001_child", "Child migration", _noop, depends_on=("6000_base",)))

    engine = _build_temp_engine()

    # Both should apply in order (6000 first since 6001 depends on it)
    applied = apply_pending_migrations(engine, registry=registry)
    assert applied == ["6000_base", "6001_child"], f"Expected ordered apply, got {applied}"


def _test_rollback_to_version() -> None:
    """Test rolling back to a specific version."""
    registry = MigrationRegistry()

    def _make_upgrade(name: str) -> MigrationFn:
        def _upgrade(engine: Engine) -> None:
            with engine.begin() as connection:
                connection.execute(text(f"CREATE TABLE {name} (id INTEGER PRIMARY KEY)"))

        return _upgrade

    def _make_downgrade(name: str) -> MigrationFn:
        def _downgrade(engine: Engine) -> None:
            with engine.begin() as connection:
                connection.execute(text(f"DROP TABLE IF EXISTS {name}"))

        return _downgrade

    registry.register(Migration("7000_a", "Table A", _make_upgrade("table_a"), _make_downgrade("table_a")))
    registry.register(Migration("7001_b", "Table B", _make_upgrade("table_b"), _make_downgrade("table_b")))
    registry.register(Migration("7002_c", "Table C", _make_upgrade("table_c"), _make_downgrade("table_c")))

    engine = _build_temp_engine()
    apply_pending_migrations(engine, registry=registry)

    # Should have all three applied
    assert set(get_applied_versions(engine)) == {"7000_a", "7001_b", "7002_c"}

    # Rollback to 7000_a (should remove 7001_b and 7002_c)
    rolled_back = rollback_to_version(engine, "7000_a", registry=registry)
    assert set(rolled_back) == {"7001_b", "7002_c"}
    assert get_applied_versions(engine) == ["7000_a"]


def _test_migration_has_rollback() -> None:
    """Test the has_rollback() method on Migration."""

    def _noop(engine: Engine) -> None:
        pass

    with_rollback = Migration("test1", "With rollback", _noop, _noop)
    without_rollback = Migration("test2", "Without rollback", _noop)

    assert with_rollback.has_rollback() is True
    assert without_rollback.has_rollback() is False


def _test_registry_get() -> None:
    """Test getting a migration by version."""
    registry = MigrationRegistry()

    def _noop(engine: Engine) -> None:
        pass

    migration = Migration("8000_get_test", "Get test", _noop)
    registry.register(migration)

    found = registry.get("8000_get_test")
    assert found is not None
    assert found.version == "8000_get_test"

    not_found = registry.get("nonexistent")
    assert not_found is None


def schema_migrator_module_tests() -> bool:
    from testing.test_framework import TestSuite, suppress_logging

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
        suite.run_test(
            "Dry-run mode previews without executing",
            _test_dry_run_mode,
            "Dry-run should report changes without applying them",
            "dry_run",
            "Run apply with dry_run=True and verify no actual changes",
        )
        suite.run_test(
            "Rollback single migration",
            _test_rollback_single_migration,
            "Should execute downgrade function and remove version record",
            "rollback",
            "Apply then rollback a migration with downgrade function",
        )
        suite.run_test(
            "Rollback fails without downgrade",
            _test_rollback_no_downgrade,
            "Should raise RollbackError when no downgrade function",
            "rollback_error",
            "Try to rollback migration without downgrade function",
        )
        suite.run_test(
            "Dependency validation",
            _test_dependency_validation,
            "Should detect invalid dependencies",
            "dependencies",
            "Register migration with nonexistent dependency and validate",
        )
        suite.run_test(
            "Dependency enforcement",
            _test_dependency_enforcement,
            "Should apply migrations in dependency order",
            "dep_order",
            "Register migrations with dependencies and verify order",
        )
        suite.run_test(
            "Rollback to specific version",
            _test_rollback_to_version,
            "Should rollback all migrations after target version",
            "rollback_to",
            "Apply multiple migrations then rollback to an earlier version",
        )
        suite.run_test(
            "Migration has_rollback method",
            _test_migration_has_rollback,
            "Should correctly report if migration has downgrade function",
            "has_rollback",
            "Check has_rollback() for migrations with and without downgrade",
        )
        suite.run_test(
            "Registry get method",
            _test_registry_get,
            "Should find migrations by version",
            "registry_get",
            "Register migration and retrieve by version string",
        )
        return suite.finish_suite()


from testing.test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(schema_migrator_module_tests)


if __name__ == "__main__":
    import os
    if os.environ.get("RUN_MODULE_TESTS") == "1":
        sys.exit(0 if run_comprehensive_tests() else 1)
    else:
        sys.exit(run_cli(sys.argv[1:]))
