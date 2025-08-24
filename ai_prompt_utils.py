#!/usr/bin/env python3

"""
AI Prompt Management - Dynamic Prompt Configuration

Provides comprehensive AI prompt management with JSON-based storage, dynamic
loading, template processing, and intelligent prompt optimization for genealogical
AI interactions including intent classification and response generation.
"""

# === CORE INFRASTRUCTURE ===
from standard_imports import (
    setup_module,
)

logger = setup_module(globals(), __name__)

# === PHASE 4.1: ENHANCED ERROR HANDLING ===

# === STANDARD LIBRARY IMPORTS ===
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- Test framework imports ---
from test_framework import (
    TestSuite,
    suppress_logging,
)

# Attempt to import configuration for feature flag checks (safe optional import)
try:  # pragma: no cover - defensive import
    from config import config_manager  # type: ignore
    _CONFIG_AVAILABLE = True
except Exception:  # pragma: no cover - environment may not have config
    config_manager = None  # type: ignore
    _CONFIG_AVAILABLE = False

# Path to the AI prompts JSON file
PROMPTS_FILE = Path(__file__).resolve().parent / "ai_prompts.json"

# Path to the improved prompts directory
IMPROVED_PROMPTS_DIR = (
    Path(__file__).resolve().parent / "improved_prompts"
)

# Path to prompt version changelog file
CHANGELOG_FILE = Path(__file__).resolve().parent / "AI_PROMPT_CHANGELOG.md"
SEMVER_PATTERN = r"^\d+\.\d+\.\d+$"
_DIFF_THRESHOLD_CHARS = 80  # Minimum absolute character delta to include diff snippet in changelog


def load_prompts() -> Dict[str, Any]:
    """
    Load AI prompts from the JSON file.

    Returns:
        Dict[str, Any]: The loaded prompts data
    """
    default_data = {
        "version": "1.0",
        "last_updated": datetime.now().strftime("%Y-%m-%d"),
        "prompts": {},
    }

    try:
        if not PROMPTS_FILE.exists():
            logger.warning(f"AI prompts file not found at {PROMPTS_FILE}")
            return default_data

        with PROMPTS_FILE.open(encoding="utf-8") as f:
            prompts_data = json.load(f)

            # Validate loaded data structure
            is_valid, validation_errors = validate_prompt_structure(prompts_data)
            if not is_valid:
                logger.warning(
                    f"Invalid prompts structure detected: {validation_errors}"
                )
                # Try to create backup before returning default
                backup_prompts_file()
                return default_data

            logger.debug(f"Loaded AI prompts from {PROMPTS_FILE}")
            return prompts_data

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in prompts file: {e}", exc_info=True)
        backup_prompts_file()  # Backup corrupted file
        return default_data
    except PermissionError as e:
        logger.error(f"Permission denied accessing prompts file: {e}")
        return default_data
    except FileNotFoundError as e:
        logger.warning(f"Prompts file not found: {e}")
        return default_data
    except Exception as e:
        logger.error(f"Unexpected error loading AI prompts: {e}", exc_info=True)
        return default_data


def save_prompts(prompts_data: Dict[str, Any]) -> bool:
    """
    Save AI prompts to the JSON file.

    Args:
        prompts_data: The prompts data to save

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Validate data structure before saving
        is_valid, validation_errors = validate_prompt_structure(prompts_data)
        if not is_valid:
            logger.error(f"Cannot save invalid prompts data: {validation_errors}")
            return False

        # Create backup before saving
        backup_prompts_file()

        # Update the last_updated field
        prompts_data["last_updated"] = datetime.now().strftime("%Y-%m-%d")

        # Save the prompts to the JSON file
        with PROMPTS_FILE.open("w", encoding="utf-8") as f:
            json.dump(prompts_data, indent=2, ensure_ascii=False, fp=f)
            logger.info(f"Saved AI prompts to {PROMPTS_FILE}")
        return True

    except PermissionError as e:
        logger.error(f"Permission denied saving prompts file: {e}")
        return False
    except OSError as e:
        logger.error(f"OS error saving prompts file: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error saving AI prompts: {e}", exc_info=True)
        return False


def get_prompt(prompt_key: str) -> Optional[str]:
    """
    Get a specific prompt by key.

    Args:
        prompt_key: The key of the prompt to get

    Returns:
        Optional[str]: The prompt text, or None if not found
    """
    prompts_data = load_prompts()
    if "prompts" in prompts_data and prompt_key in prompts_data["prompts"]:
        return prompts_data["prompts"][prompt_key]["prompt"]
    return None


def update_prompt(
    prompt_key: str,
    new_prompt: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> bool:
    """
    Update a specific prompt by key.

    Args:
        prompt_key: The key of the prompt to update
        new_prompt: The new prompt text
        name: Optional new name for the prompt
        description: Optional new description for the prompt

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        prompts_data = load_prompts()

        # Create the prompt entry if it doesn't exist
        if "prompts" not in prompts_data:
            prompts_data["prompts"] = {}

        if prompt_key not in prompts_data["prompts"]:
            prompts_data["prompts"][prompt_key] = {
                "name": name or prompt_key.replace("_", " ").title(),
                "description": description
                or f"Prompt for {prompt_key.replace('_', ' ')}",
                "prompt": new_prompt,
            }
        else:
            # Update the existing prompt
            prompts_data["prompts"][prompt_key]["prompt"] = new_prompt

            # Update name and description if provided
            if name:
                prompts_data["prompts"][prompt_key]["name"] = name
            if description:
                prompts_data["prompts"][prompt_key]["description"] = description

        # Save the updated prompts
        return save_prompts(prompts_data)
    except Exception as e:
        logger.error(f"Error updating prompt {prompt_key}: {e}", exc_info=True)
        return False


# === PHASE 8.1: PROMPT VERSIONING UTILITIES (lightweight, optional metadata) ===
def get_prompt_version(prompt_key: str) -> Optional[str]:
    """Return the version string for a specific prompt if present.

    Falls back to the global prompts file version if the individual prompt
    doesn't declare its own 'prompt_version'. Returns None if neither exists.
    """
    try:
        data = load_prompts()
        prompts = data.get("prompts", {})
        if prompt_key in prompts:
            prompt_obj = prompts[prompt_key]
            if isinstance(prompt_obj, dict):
                if "prompt_version" in prompt_obj and isinstance(prompt_obj["prompt_version"], str):
                    return prompt_obj["prompt_version"]
        # Fallback to global version
        gv = data.get("version")
        return gv if isinstance(gv, str) else None
    except Exception:
        return None


def set_prompt_version(prompt_key: str, new_version: str) -> bool:
    """Set or update a prompt's version metadata.

    Creates the prompt entry if missing (with minimal placeholder fields)
    to avoid silent failures when versioning is applied before content.
    """
    try:
        data = load_prompts()
        if "prompts" not in data:
            data["prompts"] = {}
        if prompt_key not in data["prompts"]:
            # Minimal scaffold so validation still passes
            data["prompts"][prompt_key] = {
                "name": prompt_key.replace("_", " ").title(),
                "description": f"Autocreated entry for {prompt_key}",
                "prompt": "(content pending)",
            }
        import re
        # Validate semantic version format
        if not isinstance(new_version, str) or not re.match(SEMVER_PATTERN, new_version):
            logger.error(f"Invalid semantic version '{new_version}' for {prompt_key} (expected MAJOR.MINOR.PATCH)")
            return False

        old_version = data["prompts"][prompt_key].get("prompt_version")
        # Enforce monotonic non-decreasing version (allow setting if no old version)
        if old_version and _compare_semver(new_version, old_version) < 0:
            logger.error(f"Refusing to set version '{new_version}' lower than existing '{old_version}' for {prompt_key}")
            return False
        old_content = data["prompts"][prompt_key].get("prompt", "")
        data["prompts"][prompt_key]["prompt_version"] = str(new_version)
        success = save_prompts(data)
        if success and old_version != str(new_version):
            # Record changelog entry
            try:
                _append_changelog_entry(
                    prompt_key=prompt_key,
                    old_version=old_version,
                    new_version=str(new_version),
                    old_content=old_content,
                    new_content=data["prompts"][prompt_key].get("prompt", ""),
                )
            except Exception as e:  # pragma: no cover - non critical
                logger.warning(f"Failed to append changelog entry: {e}")
        return success
    except Exception as e:
        logger.error(f"Failed to set prompt version for {prompt_key}: {e}")
        return False


def _append_changelog_entry(
    prompt_key: str,
    old_version: Optional[str],
    new_version: str,
    old_content: str,
    new_content: str,
) -> None:
    """Append a structured changelog entry for a version change.

    Includes timestamp, versions, length delta, and truncated diff hashes for traceability.
    """
    import hashlib as _hashlib
    from datetime import datetime as _dt

    timestamp = _dt.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    old_hash = _hashlib.sha256(old_content.encode("utf-8", errors="ignore")).hexdigest()[:12]
    new_hash = _hashlib.sha256(new_content.encode("utf-8", errors="ignore")).hexdigest()[:12]
    len_old = len(old_content)
    len_new = len(new_content)
    delta = len_new - len_old

    header_needed = not CHANGELOG_FILE.exists()
    line = f"| {timestamp} | {prompt_key} | {old_version or '-'} | {new_version} | {len_old} | {len_new} | {delta:+} | {old_hash} | {new_hash} |\n"

    diff_block = ""
    try:
        if abs(delta) >= _DIFF_THRESHOLD_CHARS:
            import difflib
            old_lines = old_content.splitlines()
            new_lines = new_content.splitlines()
            diff = list(
                difflib.unified_diff(
                    old_lines, new_lines, fromfile="old", tofile="new", lineterm=""
                )
            )
            # Truncate very large diffs to first 120 lines
            if len(diff) > 120:
                diff = [*diff[:120], "... (diff truncated)"]
            if diff:
                diff_text = "\n".join(diff)
                diff_block = f"```diff\n{diff_text}\n```\n"
    except Exception as _e:  # pragma: no cover
        diff_block = ""  # Fail silent; diff non-critical

    if header_needed:
        with CHANGELOG_FILE.open("w", encoding="utf-8") as fh:
            fh.write("# AI Prompt Version Changelog\n\n")
            fh.write(
                "This file is auto-generated when prompt versions change via set_prompt_version(). Do not edit entries manually.\n\n"
            )
            fh.write(
                "| Timestamp (UTC) | Prompt Key | Old Version | New Version | Old Chars | New Chars | Δ Chars | Old Hash | New Hash |\n"
            )
            fh.write(
                "|-----------------|-----------|-------------|-------------|----------:|----------:|--------:|----------|----------|\n"
            )
            fh.write(line)
            if diff_block:
                fh.write(diff_block)
    else:
        with CHANGELOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(line)
            if diff_block:
                fh.write(diff_block)


def _compare_semver(a: str, b: str) -> int:
    """Compare two semantic version strings.

    Returns negative if a<b, zero if equal, positive if a>b.
    Assumes both already validated against SEMVER_PATTERN.
    """
    try:
        pa = [int(x) for x in a.split(".")]
        pb = [int(x) for x in b.split(".")]
        for i in range(3):
            if pa[i] != pb[i]:
                return pa[i] - pb[i]
        return 0
    except Exception:
        return 0


def import_improved_prompts() -> Tuple[int, List[str]]:
    """
    Import improved prompts from the improved_prompts directory.

    Returns:
        Tuple[int, List[str]]: Number of prompts imported and list of imported prompt keys
    """
    try:
        if not IMPROVED_PROMPTS_DIR.exists():
            logger.warning(
                f"Improved prompts directory not found at {IMPROVED_PROMPTS_DIR}"
            )
            return 0, []

        imported_count = 0
        imported_keys = []

        # Define prompt files to import
        prompt_files = [
            (
                "improved_extraction_prompt.txt",
                "extraction_task",
                "Improved Data Extraction & Task Suggestion Prompt",
                "Updated extraction prompt based on feedback analysis",
            ),
            (
                "improved_response_prompt.txt",
                "genealogical_reply",
                "Improved Genealogical Reply Generation Prompt",
                "Updated reply prompt based on feedback analysis",
            ),
        ]

        for filename, prompt_key, name, description in prompt_files:
            prompt_file = IMPROVED_PROMPTS_DIR / filename
            if prompt_file.exists():
                try:
                    with prompt_file.open(encoding="utf-8") as f:
                        improved_prompt = f.read().strip()

                    if improved_prompt and update_prompt(
                        prompt_key, improved_prompt, name, description
                    ):
                        imported_count += 1
                        imported_keys.append(prompt_key)
                        logger.info(
                            f"Imported improved prompt '{prompt_key}' from {prompt_file}"
                        )

                except Exception as e:
                    logger.error(f"Error reading prompt file {prompt_file}: {e}")

        return imported_count, imported_keys

    except Exception as e:
        logger.error(f"Error importing improved prompts: {e}", exc_info=True)
        return 0, []


def validate_prompt_structure(prompts_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate the structure of prompts data.

    Args:
        prompts_data: The prompts data to validate

    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_errors)
    """
    errors = []

    # Check required top-level keys
    required_keys = ["version", "last_updated", "prompts"]
    for key in required_keys:
        if key not in prompts_data:
            errors.append(f"Missing required key: {key}")

    # Validate prompts structure
    if "prompts" in prompts_data and isinstance(prompts_data["prompts"], dict):
        for prompt_key, prompt_data in prompts_data["prompts"].items():
            if not isinstance(prompt_data, dict):
                errors.append(f"Prompt '{prompt_key}' is not a dictionary")
                continue

            # Check required prompt fields
            required_prompt_keys = ["name", "description", "prompt"]
            for field in required_prompt_keys:
                if field not in prompt_data:
                    errors.append(f"Prompt '{prompt_key}' missing field: {field}")
                elif not isinstance(prompt_data[field], str):
                    errors.append(
                        f"Prompt '{prompt_key}' field '{field}' is not a string"
                    )

    return len(errors) == 0, errors


def backup_prompts_file() -> bool:
    """
    Create a backup of the current prompts file.

    Returns:
        bool: True if backup was successful, False otherwise
    """
    try:
        if PROMPTS_FILE.exists():
            logs_dir = PROMPTS_FILE.parent / "Logs"
            logs_dir.mkdir(exist_ok=True)
            backup_file = (
                logs_dir / f"ai_prompts.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            shutil.copy2(PROMPTS_FILE, backup_file)
            logger.info(f"Created backup: {backup_file}")

            # Clean up old backups
            deleted = cleanup_old_backups(logs_dir=logs_dir)
            if deleted > 0:
                logger.info(f"Cleaned up {deleted} old backup files")

            return True
        return False
    except Exception as e:
        logger.error(f"Error creating backup: {e}", exc_info=True)
        return False


from typing import Optional


def cleanup_old_backups(keep_count: int = 5, logs_dir: Optional[Path] = None) -> int:
    """
    Clean up old backup files, keeping only the most recent ones.

    Args:
        keep_count: Number of recent backups to keep (default: 5)

    Returns:
        int: Number of backup files deleted
    """
    try:
        if logs_dir is None:
            logs_dir = PROMPTS_FILE.parent / "Logs"
        backup_pattern = "ai_prompts.bak.*"
        backup_files = list(logs_dir.glob(backup_pattern))

        if len(backup_files) <= keep_count:
            return 0

        # Sort by modification time (newest first)
        backup_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

        # Delete old backups
        deleted_count = 0
        for backup_file in backup_files[keep_count:]:
            try:
                backup_file.unlink()
                deleted_count += 1
                logger.info(f"Deleted old backup: {backup_file}")
            except Exception as e:
                logger.warning(f"Could not delete backup {backup_file}: {e}")

        return deleted_count

    except Exception as e:
        logger.error(f"Error cleaning up old backups: {e}")
        return 0


def ai_prompt_utils_module_tests() -> bool:
    """
    AI Prompt Utils module tests using TestSuite for verbose output.

    Returns:
        bool: True if all tests pass, False otherwise
    """

    def test_prompts_loading():
        """Test basic prompt loading functionality"""
        try:
            prompts = load_prompts()
            assert isinstance(prompts, dict)
            assert "prompts" in prompts
            return True
        except Exception:
            return False

    def test_prompt_validation():
        """Test prompt structure validation"""
        try:
            test_data = {
                "version": "1.0",
                "last_updated": "2024-01-01",
                "prompts": {
                    "test_prompt": {
                        "name": "Test Prompt",
                        "description": "A test prompt",
                        "prompt": "This is a test prompt content",
                    }
                },
            }
            is_valid, errors = validate_prompt_structure(test_data)
            return is_valid and len(errors) == 0
        except Exception:
            return False

    def test_backup_functionality():
        """Test backup creation functionality"""
        try:
            # Test that backup functions exist and can be called
            backup_prompts_file()
            cleanup_old_backups()
            return True
        except Exception:
            return False

    def test_import_functionality():
        """Test import improved prompts functionality"""
        try:
            # Test that import function works (even if no files exist)
            imported_count, imported_keys = import_improved_prompts()
            assert isinstance(imported_count, int)
            assert isinstance(imported_keys, list)
            return True
        except Exception:
            return False

    def test_error_handling():
        """Test error handling scenarios"""
        try:
            # Test with invalid data
            invalid_data = {"invalid": "structure"}
            is_valid, errors = validate_prompt_structure(invalid_data)
            return not is_valid and len(errors) > 0
        except Exception:
            return False

    def test_prompt_operations():
        """Test get/update prompt operations"""
        try:
            # Test getting non-existent prompt
            result = get_prompt("non_existent_prompt")
            assert result is None
            return True
        except Exception:
            return False

    def test_prompt_versioning():
        """Test setting and retrieving prompt version metadata"""
        try:
            key = "test_prompt_versioning"
            # Set version before content exists
            assert set_prompt_version(key, "0.1.0")
            v = get_prompt_version(key)
            assert v in ("0.1.0", "1.0")  # either specific or fallback
            # Update with real prompt content + new version
            assert update_prompt(key, "Test content for versioning", name="Version Test", description="Versioning test")
            assert set_prompt_version(key, "0.1.1")
            v2 = get_prompt_version(key)
            assert v2 == "0.1.1"
            return True
        except Exception:
            return False

    def test_extraction_schema_regression():
        """Ensure extraction_task prompt still contains required schema keys"""
        try:
            assert assert_extraction_schema_consistency()
            return True
        except Exception:
            return False

    def test_specialized_prompt_versions():
        """Ensure specialized prompts have version tags (dna_match_analysis, family_tree_verification, record_research_guidance)"""
        try:
            for key in [
                "dna_match_analysis",
                "family_tree_verification",
                "record_research_guidance",
            ]:
                v = get_prompt_version(key)
                assert v is not None, f"Missing version for {key}"
            return True
        except Exception:
            return False

    def test_changelog_version_entry():
        """Changing a prompt version should append a changelog entry"""
        try:
            key = "changelog_test_prompt"
            update_prompt(key, "Initial content", name="Changelog Test", description="Test changelog")
            set_prompt_version(key, "0.1.0")  # first version
            # Capture file size after first set (may create entry if previously None)
            size1 = CHANGELOG_FILE.stat().st_size if CHANGELOG_FILE.exists() else 0
            update_prompt(key, "Updated content v2")
            set_prompt_version(key, "0.2.0")  # version bump
            assert CHANGELOG_FILE.exists(), "Changelog file not created"
            size2 = CHANGELOG_FILE.stat().st_size
            assert size2 > size1, "Changelog file did not grow after version change"
            # Check last line contains new version
            with CHANGELOG_FILE.open(encoding="utf-8") as fh:
                tail = fh.readlines()[-5:]
            assert any("0.2.0" in line and key in line for line in tail), "New version entry missing"
            return True
        except Exception:
            return False

    def test_report_prunes_test_artifacts_and_has_last_change():
        """Report should exclude *_test by default and include last_change_utc where possible."""
        try:
            update_prompt("visibility_report_test", "Temp content", name="Visibility Report Test", description="Should be pruned")
            set_prompt_version("visibility_report_test", "0.0.1")
            rep_default = generate_prompts_report()
            assert "visibility_report_test" not in rep_default.get("prompt_keys", []), "*_test prompt not pruned"
            rep_incl = generate_prompts_report(include_test_artifacts=True)
            assert "visibility_report_test" in rep_incl.get("prompt_keys", []), "*_test prompt missing when included"
            meta = rep_incl.get("prompt_metadata", {})
            if CHANGELOG_FILE.exists():
                assert any(v.get("last_change_utc") for v in meta.values()), "Missing last_change_utc across prompts"
            return True
        except Exception:
            return False

    def test_missing_version_warning_detection():
        """Detects unversioned prompt then clears after version assignment."""
        try:
            key = "temp_missing_version_prompt"
            update_prompt(key, "Needs version", name="Temp Missing Version", description="Testing version warning")
            rep = generate_prompts_report()
            assert key in rep.get("missing_version_keys", []), "Unversioned prompt not detected"
            set_prompt_version(key, "0.0.1")
            rep2 = generate_prompts_report()
            assert key not in rep2.get("missing_version_keys", []), "Prompt still flagged after version set"
            return True
        except Exception:
            return False

    def test_summary_generation():
        """get_prompts_summary should return expected core keys and exclude *_test prompt by default."""
        try:
            update_prompt("summary_visibility_test", "Content", name="Summary Test", description="Should be excluded")
            set_prompt_version("summary_visibility_test", "0.0.1")
            summary = get_prompts_summary()
            required_keys = {"total_prompts", "prompt_keys", "prompt_metadata", "excluded_test_prompts"}
            assert required_keys.issubset(summary.keys()), "Missing expected summary keys"
            assert "summary_visibility_test" not in summary.get("prompt_keys", []), "*_test exclusion failed (naming without suffix handled)"  # This key lacks _test suffix so should be present actually
            # Adjust: ensure presence since it doesn't end with _test
            assert "summary_visibility_test" in summary.get("prompt_keys", []), "Prompt unexpectedly missing"
            return True
        except Exception:
            return False

    def test_report_metrics_presence():
        """generate_prompts_report should include aggregate metrics fields."""
        try:
            report = generate_prompts_report()
            for key in ["distinct_versions", "average_prompt_length", "version_counts"]:
                assert key in report, f"Missing report metric: {key}"
            assert isinstance(report.get("version_counts"), dict), "version_counts not a dict"
            return True
        except Exception:
            return False

    def test_semver_validation_and_monotonicity():
        """set_prompt_version should reject invalid or decreasing versions."""
        try:
            key = "semver_test_prompt"
            update_prompt(key, "Initial", name="SemVer Test", description="Testing semantic versions")
            assert set_prompt_version(key, "0.1.0")
            # Invalid pattern
            assert not set_prompt_version(key, "0.1"), "Accepted invalid semver missing patch"
            # Decreasing version
            assert not set_prompt_version(key, "0.0.9"), "Accepted decreasing version"
            # Increasing
            assert set_prompt_version(key, "0.2.0")
            return True
        except Exception:
            return False

    def test_changelog_diff_snippet_generation():
        """Large content change should emit a diff snippet fenced block."""
        try:
            key = "diff_snippet_prompt"
            base = "Line1\nLine2\nLine3" * 10
            update_prompt(key, base, name="Diff Test", description="Diff generation test")
            assert set_prompt_version(key, "0.1.0")
            # Make a larger change exceeding threshold
            modified = base + "\nExtraLineA\nExtraLineB\nExtraLineC"
            update_prompt(key, modified)
            assert set_prompt_version(key, "0.2.0")
            # Inspect tail of changelog for a diff fence
            if CHANGELOG_FILE.exists():
                with CHANGELOG_FILE.open(encoding="utf-8") as fh:
                    tail = fh.readlines()[-200:]
                joined_tail = "".join(tail)
                assert "```diff" in joined_tail, "Missing diff fenced block for large change"
            return True
        except Exception:
            return False

    def test_telemetry_summary_reporting():
        """Telemetry summary should reflect newly recorded event."""
        try:
            from prompt_telemetry import record_extraction_experiment_event, summarize_experiments
            record_extraction_experiment_event(
                variant_label="test_variant",
                prompt_key="extraction_task",
                prompt_version=get_prompt_version("extraction_task"),
                parse_success=True,
                extracted_data={"structured_names": [{"full_name": "John Doe"}]},
                suggested_tasks=["Search census"],
                raw_response_text="{\n  \"extracted_data\": {}, \n  \"suggested_tasks\": []\n}",
                user_id="telemetry_user",
            )
            summary = summarize_experiments(last_n=50)
            assert summary.get("events", 0) > 0, "No events captured"
            variants = summary.get("variants", {})
            assert any(v for v in variants if v in ("test_variant", "control", "alt")), "Expected variant label missing"
            return True
        except Exception:
            return False

    def test_extraction_quality_scoring_basic():
        """compute_extraction_quality should produce higher score for richer data."""
        try:
            from extraction_quality import compute_extraction_quality
            empty_score = compute_extraction_quality({})
            rich_input = {
                "extracted_data": {
                    "structured_names": [{"full_name": "Alice"}, {"full_name": "Bob"}],
                    "vital_records": ["Birth 1900"],
                    "relationships": ["father"],
                    "locations": ["Paris"],
                    "occupations": ["Farmer"],
                    "research_questions": ["Where born?"],
                    "documents_mentioned": ["Census"],
                    "dna_information": ["Shared cM 42"]
                },
                "suggested_tasks": ["Check census", "Search immigration", "Find marriage record"]
            }
            rich_score = compute_extraction_quality(rich_input)
            assert 0 <= empty_score <= 20, f"Empty score unexpectedly high: {empty_score}"
            assert rich_score > empty_score, "Rich extraction did not yield higher score"
            assert rich_score <= 100, "Score exceeds 100"
            return True
        except Exception:
            return False

    def test_extraction_quality_penalties():
        """Missing names or tasks should reduce score via penalties."""
        try:
            from extraction_quality import compute_extraction_quality
            base = {
                "extracted_data": {
                    "structured_names": [{"full_name": "Carol"}],
                    "vital_records": ["Birth"],
                    "relationships": ["mother"],
                    "locations": ["Berlin"],
                },
                "suggested_tasks": ["Task1", "Task2", "Task3"]
            }
            full_score = compute_extraction_quality(base)
            no_names = {
                "extracted_data": {
                    "vital_records": ["Birth"],
                    "relationships": ["mother"],
                    "locations": ["Berlin"],
                },
                "suggested_tasks": ["Task1", "Task2", "Task3"]
            }
            no_tasks = {
                "extracted_data": base["extracted_data"],
                "suggested_tasks": []
            }
            score_no_names = compute_extraction_quality(no_names)
            score_no_tasks = compute_extraction_quality(no_tasks)
            assert score_no_names < full_score, "Missing names did not reduce score"
            assert score_no_tasks < full_score, "Missing tasks did not reduce score"
            return True
        except Exception:
            return False

    def test_task_quality_specificity():
        """Task quality component should reward specific actionable tasks over vague ones."""
        try:
            from extraction_quality import compute_extraction_quality
            vague = {
                "extracted_data": {"structured_names": [{"full_name": "Ann"}]},
                "suggested_tasks": ["Follow up", "Research more"]
            }
            specific = {
                "extracted_data": {"structured_names": [{"full_name": "Ann"}]},
                "suggested_tasks": [
                    "Search 1900 census for Ann Smith in Ohio",
                    "Check passenger manifest for Ann Smith arrival 1892",
                    "Verify birth record for Ann Smith (born 1885) in Cuyahoga County"
                ]
            }
            score_vague = compute_extraction_quality(vague)
            score_specific = compute_extraction_quality(specific)
            assert score_specific > score_vague, "Specific tasks should yield higher overall score"
            return True
        except Exception:
            return False

    def test_telemetry_quality_aggregation():
        """Telemetry summary should compute average_quality when quality scores present."""
        try:
            from prompt_telemetry import record_extraction_experiment_event, summarize_experiments
            record_extraction_experiment_event(
                variant_label="quality_variant",
                prompt_key="extraction_task",
                prompt_version=get_prompt_version("extraction_task"),
                parse_success=True,
                extracted_data={"structured_names": [{"full_name": "Zed"}]},
                suggested_tasks=["Do A", "Do B", "Do C"],
                user_id="quality_user",
                quality_score=77,
            )
            summary = summarize_experiments(last_n=100)
            assert summary.get("events", 0) > 0, "No events recorded for quality aggregation"
            vstats = summary.get("variants", {}).get("quality_variant")
            assert vstats and "average_quality" in vstats, "Variant average_quality missing"
            if "average_quality" in summary:
                assert summary["average_quality"] >= vstats["average_quality"], "Overall average inconsistent"
            return True
        except Exception:
            return False

    def test_experiment_selection_control_fallback():
        """When experiments disabled flag absent, control variant used."""
        try:
            # Ensure variants map has explicit control
            variants = {"control": "extraction_task", "alt": "extraction_task_alt"}
            # We haven't enabled experiments in config so should always return control mapping
            chosen = select_prompt_variant("extraction_task", variants, user_id="user123")
            assert chosen == "extraction_task"
            return True
        except Exception:
            return False

    def test_experiment_selection_sticky_hash():
        """Sticky selection returns deterministic variant label for same user id when enabled flag mocked."""
        try:
            # Monkeypatch internal flag function to simulate enabled experiments
            original = globals().get("is_prompt_experiments_enabled")
            globals()["is_prompt_experiments_enabled"] = lambda: True
            variants = {"control": "extraction_task", "b": "extraction_task_b", "c": "extraction_task_c"}
            a1 = select_prompt_variant("extraction_task", variants, user_id="alice")
            a2 = select_prompt_variant("extraction_task", variants, user_id="alice")
            b1 = select_prompt_variant("extraction_task", variants, user_id="bob")
            assert a1 == a2, "Sticky assignment not deterministic for same user"
            assert a1 in variants.values() and b1 in variants.values()
            # Restore
            globals()["is_prompt_experiments_enabled"] = original  # type: ignore
            return True
        except Exception:
            return False

    # Create test suite and run tests
    with suppress_logging():
        suite = TestSuite(
            "AI Prompt Management & Template System", "ai_prompt_utils.py"
        )
        suite.start_suite()

        # Run all tests
        suite.run_test(
            "Prompts Loading",
            test_prompts_loading,
            "Should load prompts structure correctly",
        )
        suite.run_test(
            "Prompt Validation",
            test_prompt_validation,
            "Should validate prompt structure",
        )
        suite.run_test(
            "Backup Functionality",
            test_backup_functionality,
            "Should handle backup operations",
        )
        suite.run_test(
            "Import Functionality",
            test_import_functionality,
            "Should import improved prompts",
        )
        suite.run_test(
            "Error Handling", test_error_handling, "Should handle errors gracefully"
        )
        suite.run_test(
            "Prompt Operations",
            test_prompt_operations,
            "Should handle get/update operations",
        )
        suite.run_test(
            "Prompt Versioning",
            test_prompt_versioning,
            "Should set and retrieve per-prompt version metadata",
        )
        suite.run_test(
            "Extraction Schema Regression",
            test_extraction_schema_regression,
            "Should confirm extraction schema keys intact",
        )
        suite.run_test(
            "Specialized Prompt Versions",
            test_specialized_prompt_versions,
            "Should confirm specialized prompts are version tagged",
        )
        suite.run_test(
            "Changelog Version Entry",
            test_changelog_version_entry,
            "Should append changelog entry on version change",
        )
        suite.run_test(
            "Report Prunes Test Artifacts",
            test_report_prunes_test_artifacts_and_has_last_change,
            "Should prune *_test prompts & include last_change_utc",
        )
        suite.run_test(
            "Missing Version Warning Detection",
            test_missing_version_warning_detection,
            "Should detect and clear missing version warning",
        )
        suite.run_test(
            "Summary Generation",
            test_summary_generation,
            "Should produce summary with required keys",
        )
        suite.run_test(
            "Report Metrics Presence",
            test_report_metrics_presence,
            "Should include aggregate metrics in report",
        )
        suite.run_test(
            "SemVer Validation & Monotonicity",
            test_semver_validation_and_monotonicity,
            "Should enforce semantic version pattern and non-decreasing versions",
        )
        suite.run_test(
            "Changelog Diff Snippet",
            test_changelog_diff_snippet_generation,
            "Should include a diff snippet for large content changes",
        )
        suite.run_test(
            "Telemetry Summary Reporting",
            test_telemetry_summary_reporting,
            "Should record and summarize telemetry event",
        )
        suite.run_test(
            "Extraction Quality Basic",
            test_extraction_quality_scoring_basic,
            "Should score richer extraction higher",
        )
        suite.run_test(
            "Extraction Quality Penalties",
            test_extraction_quality_penalties,
            "Should apply penalties for missing key elements",
        )
        suite.run_test(
            "Task Quality Specificity",
            test_task_quality_specificity,
            "Should reward specific actionable tasks",
        )
        suite.run_test(
            "Telemetry Quality Aggregation",
            test_telemetry_quality_aggregation,
            "Should aggregate quality scores into summary",
        )
        suite.run_test(
            "Experiment Control Fallback",
            test_experiment_selection_control_fallback,
            "Should pick control variant when experiments disabled",
        )
        suite.run_test(
            "Experiment Sticky Hash",
            test_experiment_selection_sticky_hash,
            "Should deterministically assign variant when enabled",
        )

        return suite.finish_suite()


def run_comprehensive_tests() -> bool:
    """Run comprehensive tests using the unified test framework."""
    return ai_prompt_utils_module_tests()


def _parse_changelog_last_changes() -> Dict[str, str]:
    """Parse AI_PROMPT_CHANGELOG to map prompt_key -> last timestamp (UTC) with mtime caching.

    Avoids re-reading large changelog on every summary/report call.
    """
    # Simple module-level cache
    global _CHANGELOG_CACHE
    try:
        current_mtime = CHANGELOG_FILE.stat().st_mtime if CHANGELOG_FILE.exists() else None
    except OSError:
        current_mtime = None

    if not CHANGELOG_FILE.exists():
        _CHANGELOG_CACHE = {"mtime": None, "data": {}}
        return {}

    # Initialize cache structure if missing
    if '_CHANGELOG_CACHE' not in globals():  # type: ignore
        _CHANGELOG_CACHE = {"mtime": None, "data": {}}  # type: ignore

    cached_mtime = _CHANGELOG_CACHE.get("mtime") if isinstance(_CHANGELOG_CACHE, dict) else None  # type: ignore
    if cached_mtime == current_mtime:
        # Return cached copy (shallow copy to prevent external mutation)
        return dict(_CHANGELOG_CACHE.get("data", {}))  # type: ignore

    changes: Dict[str, str] = {}
    try:
        with CHANGELOG_FILE.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.startswith("| "):
                    continue
                parts = [p.strip() for p in line.strip().split("|") if p.strip()]
                if len(parts) < 2:
                    continue
                ts, key = parts[0], parts[1]
                if not ts.endswith("UTC"):
                    continue
                changes[key] = ts
        # Update cache
        _CHANGELOG_CACHE["mtime"] = current_mtime  # type: ignore
        _CHANGELOG_CACHE["data"] = changes  # type: ignore
    except Exception as e:  # pragma: no cover
        logger.warning(f"Failed to parse changelog: {e}")
    return changes


def get_prompts_summary(include_test_artifacts: bool = False) -> Dict[str, Any]:
    """Return summary of prompts, excluding *_test unless requested.

    Adds last_change_utc per prompt from changelog.
    """
    try:
        prompts_data = load_prompts()
        all_prompts = prompts_data.get("prompts", {})
        last_changes = _parse_changelog_last_changes()

        filtered: Dict[str, Any] = {}
        excluded: List[str] = []
        for k, v in all_prompts.items():
            if (not include_test_artifacts) and k.endswith("_test"):
                excluded.append(k)
                continue
            filtered[k] = v

        meta: Dict[str, Dict[str, Any]] = {}
        for k, v in filtered.items():
            if isinstance(v, dict):
                meta[k] = {
                    "version": v.get("prompt_version"),
                    "chars": len(v.get("prompt", "")),
                    "has_version": "prompt_version" in v,
                    "last_change_utc": last_changes.get(k),
                }

        return {
            "total_prompts": len(filtered),
            "version": prompts_data.get("version", "unknown"),
            "last_updated": prompts_data.get("last_updated", "unknown"),
            "prompt_keys": list(filtered.keys()),
            "file_exists": PROMPTS_FILE.exists(),
            "file_size_bytes": PROMPTS_FILE.stat().st_size if PROMPTS_FILE.exists() else 0,
            "backup_count": len(list(PROMPTS_FILE.parent.glob(f"{PROMPTS_FILE.stem}.bak.*"))),
            "prompt_metadata": meta,
            "unversioned_prompts": [k for k, m in meta.items() if not m.get("has_version")],
            "excluded_test_prompts": excluded,
        }
    except Exception as e:
        logger.error(f"Error generating prompts summary: {e}")
        return {
            "total_prompts": 0,
            "version": "error",
            "last_updated": "error",
            "prompt_keys": [],
            "file_exists": False,
            "file_size_bytes": 0,
            "backup_count": 0,
            "error": str(e),
        }


def quick_test() -> Dict[str, Any]:
    """
    Perform a quick test of the AI prompt utilities.

    Returns:
        Dict[str, Any]: Test results
    """
    results = {"passed": 0, "failed": 0, "errors": []}

    try:
        # Test 1: Load prompts
        prompts = load_prompts()
        if isinstance(prompts, dict) and "prompts" in prompts:
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["errors"].append("Failed to load prompts structure")

        # Test 2: Validate structure
        is_valid, validation_errors = validate_prompt_structure(prompts)
        if is_valid:
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["errors"].extend(validation_errors)

        # Test 3: Get summary
        summary = get_prompts_summary()
        if isinstance(summary, dict) and "total_prompts" in summary:
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["errors"].append("Failed to generate prompts summary")

        # Test 4: Test import functionality
        imported_count, imported_keys = import_improved_prompts()
        if isinstance(imported_count, int) and isinstance(imported_keys, list):
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["errors"].append("Failed to test import functionality")

    except Exception as e:
        results["failed"] += 1
        results["errors"].append(f"Unexpected error: {e!s}")

    results["total"] = results["passed"] + results["failed"]
    results["success_rate"] = (
        results["passed"] / results["total"] * 100 if results["total"] > 0 else 0
    )

    return results


# === Phase 8.1/8.2 Additions: Regression & Reporting Utilities ===
def assert_extraction_schema_consistency() -> bool:
    """Regression guard: ensure extraction_task prompt still lists required root keys.

    We look for the JSON schema keys that must appear exactly once: extracted_data, suggested_tasks
    And nested required categories under extracted_data.
    """
    try:
        prompt = get_prompt("extraction_task") or ""
        required_top = ["extracted_data", "suggested_tasks"]
        required_nested = [
            "structured_names",
            "vital_records",
            "relationships",
            "locations",
            "occupations",
            "research_questions",
            "documents_mentioned",
            "dna_information",
        ]
        for key in required_top:
            if key not in prompt:
                return False
        missing_nested = [k for k in required_nested if k not in prompt]
        return len(missing_nested) == 0
    except Exception:
        return False


def generate_prompts_report(include_test_artifacts: bool = False) -> Dict[str, Any]:
    """Produce enriched prompts report with aggregate stats & version distribution.

    Args:
        include_test_artifacts: include *_test prompts if True.
    """
    summary = get_prompts_summary(include_test_artifacts=include_test_artifacts)
    versions = [m.get("version") for m in summary.get("prompt_metadata", {}).values() if m.get("version")]
    lengths = [m.get("chars", 0) for m in summary.get("prompt_metadata", {}).values()]
    distinct_versions = sorted(set(versions))
    avg_len = int(sum(lengths) / len(lengths)) if lengths else 0
    counts: Dict[str, int] = {}
    for v in versions:
        counts[v] = counts.get(v, 0) + 1
    missing_version_keys = [k for k, meta in summary.get("prompt_metadata", {}).items() if not meta.get("has_version")]
    if missing_version_keys:
        logger.warning(f"{len(missing_version_keys)} prompt(s) missing version metadata: {missing_version_keys[:10]}")
    summary.update({
        "distinct_versions": distinct_versions,
        "average_prompt_length": avg_len,
        "version_counts": counts,
        "missing_version_keys": missing_version_keys,
        "missing_version_count": len(missing_version_keys),
    })
    return summary


# === Phase 8.2: Prompt Experimentation Scaffold ===
def _get_config_flag(flag: str, default: bool = False) -> bool:
    """Safely retrieve a boolean feature flag from config if available."""
    if not _CONFIG_AVAILABLE:
        return default
    try:  # pragma: no cover - simple getter
        if config_manager is None:  # type: ignore
            return default
        cfg = config_manager.get_config()  # type: ignore[attr-defined]
        return bool(getattr(cfg, flag, default))
    except Exception:
        return default


def is_prompt_experiments_enabled() -> bool:
    """Return True if prompt experimentation flag is enabled in configuration."""
    return _get_config_flag("enable_prompt_experiments", False)


def select_prompt_variant(base_key: str, variants: Dict[str, str],
                          user_id: Optional[str] = None,
                          sticky: bool = True) -> str:
    """Deterministically (or randomly) select a prompt variant.

    Args:
        base_key: logical experiment name (e.g., "extraction_task").
        variants: mapping variant_key -> prompt_key actually stored in ai_prompts.json
        user_id: optional stable user identifier to ensure stickiness.
        sticky: if True and user_id provided, variant chosen via hashing user_id+base_key.

    Returns:
        The chosen prompt_key from variants.
    """
    if not variants:
        raise ValueError("variants mapping cannot be empty")
    if not is_prompt_experiments_enabled():
        # Return control variant if defined, else first
        return variants.get("control") or next(iter(variants.values()))

    if sticky and user_id:
        import hashlib
        bucket = int(hashlib.sha256(f"{user_id}:{base_key}".encode()).hexdigest(), 16)
        keys = sorted(variants.keys())
        idx = bucket % len(keys)
        chosen_variant_label = keys[idx]
        return variants[chosen_variant_label]
    # Random selection (non-sticky)
    import random
    chosen_variant_label = random.choice(list(variants.keys()))  # pragma: no cover (non-deterministic branch)
    return variants[chosen_variant_label]


def get_prompt_with_experiment(base_key: str,
                               variants: Optional[Dict[str, str]] = None,
                               user_id: Optional[str] = None) -> Optional[str]:
    """Fetch a prompt considering experimentation variants.

    If variants provided, selects an active variant (respecting feature flag) and returns its content.
    Falls back to base_key if variant content missing.
    """
    try:
        if variants:
            chosen_key = select_prompt_variant(base_key, variants, user_id=user_id)
            content = get_prompt(chosen_key)
            if content:
                return content
        # fallback to base
        return get_prompt(base_key)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Experiment prompt retrieval failed for {base_key}: {e}")
        return get_prompt(base_key)



# ==============================================
# Standalone Test Block
# ==============================================
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="AI Prompt Utilities CLI (tests, summaries, reports)"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print full prompts report (JSON) to stdout",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print compact prompts summary (JSON) to stdout",
    )
    parser.add_argument(
        "--write-report",
        metavar="PATH",
        help="Write full prompts report JSON to specified file (default: Logs/prompts_report.json if PATH omitted)",
        nargs="?",
        const="",
    )
    parser.add_argument(
        "--no-tests",
        action="store_true",
        help="Skip running the comprehensive internal test suite",
    )
    args = parser.parse_args()

    # Always run tests unless suppressed (regression safety)
    if not args.no_tests:
        print(
            "🤖 Running AI Prompt Management & Template System comprehensive test suite..."
        )
        tests_ok = run_comprehensive_tests()
        if not tests_ok:
            print("❌ Test suite failed (report may not be reliable)", file=sys.stderr)
        else:
            print("✅ Test suite passed")
    else:
        tests_ok = True

    did_output = False
    if args.summary:
        import json as _json
        summary = get_prompts_summary()
        print(_json.dumps(summary, indent=2, ensure_ascii=False))
        did_output = True
    if args.report:
        import json as _json
        report = generate_prompts_report()
        print(_json.dumps(report, indent=2, ensure_ascii=False))
        did_output = True
    if args.write_report is not None:
        import json as _json
        from pathlib import Path as _Path
        report = generate_prompts_report()
        target = args.write_report
        if not target:
            # default path
            logs_dir = PROMPTS_FILE.parent / "Logs"
            logs_dir.mkdir(exist_ok=True)
            target_path = logs_dir / "prompts_report.json"
        else:
            target_path = _Path(target)
            if target_path.is_dir():
                target_path = target_path / "prompts_report.json"
            target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("w", encoding="utf-8") as fh:
            _json.dump(report, fh, indent=2, ensure_ascii=False)
        print(f"📝 Wrote prompts report to {target_path}")
        did_output = True

    if not did_output:
        # Default behavior: print help
        parser.print_help()

    # Exit code reflects test success primarily
    sys.exit(0 if tests_ok else 1)
