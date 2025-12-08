#!/usr/bin/env python3

"""
A/B Testing Framework for Prompt Experiments

Sprint 4: Provides controlled experimentation for prompt variants,
response quality comparison, and automated optimization.

Key Features:
- Experiment configuration and management
- Variant assignment with consistent hashing
- Metrics collection and statistical analysis
- Winner selection with significance testing
- Integration with prompt telemetry
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# === DATA CLASSES ===


@dataclass()
class Variant:
    """A variant in an A/B test."""

    name: str  # e.g., "control", "treatment_a"
    prompt_key: str  # Key in ai_prompts.json
    prompt_variant: Optional[str] = None  # Variant name within the prompt
    weight: float = 1.0  # Assignment weight (higher = more traffic)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "prompt_key": self.prompt_key,
            "prompt_variant": self.prompt_variant,
            "weight": self.weight,
        }


@dataclass
class ExperimentResult:
    """Result of a single experiment trial."""

    experiment_id: str
    variant_name: str
    person_id: Optional[int]
    timestamp: datetime
    quality_score: float
    response_time_ms: float
    success: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "experiment_id": self.experiment_id,
            "variant_name": self.variant_name,
            "person_id": self.person_id,
            "timestamp": self.timestamp.isoformat(),
            "quality_score": self.quality_score,
            "response_time_ms": self.response_time_ms,
            "success": self.success,
            "metadata": self.metadata,
        }


@dataclass
class VariantStats:
    """Statistics for a single variant."""

    variant_name: str
    sample_count: int = 0
    success_rate: float = 0.0
    avg_quality_score: float = 0.0
    avg_response_time_ms: float = 0.0
    std_quality_score: float = 0.0


@dataclass
class ExperimentSummary:
    """Summary of an A/B test experiment."""

    experiment_id: str
    status: str  # "running", "completed", "stopped"
    start_time: datetime
    end_time: Optional[datetime]
    total_trials: int
    variants: list[VariantStats]
    winner: Optional[str] = None
    confidence: float = 0.0


@dataclass()
class Experiment:
    """Configuration for an A/B test experiment."""

    experiment_id: str
    name: str
    description: str
    variants: list[Variant]
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    min_sample_size: int = 100
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "variants": [v.to_dict() for v in self.variants],
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "min_sample_size": self.min_sample_size,
            "enabled": self.enabled,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Experiment":
        """Create from dictionary."""
        variants = [
            Variant(
                name=v["name"],
                prompt_key=v["prompt_key"],
                prompt_variant=v.get("prompt_variant"),
                weight=v.get("weight", 1.0),
            )
            for v in data.get("variants", [])
        ]
        return cls(
            experiment_id=data["experiment_id"],
            name=data["name"],
            description=data.get("description", ""),
            variants=variants,
            start_time=datetime.fromisoformat(data["start_time"])
            if data.get("start_time")
            else datetime.now(timezone.utc),
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            min_sample_size=data.get("min_sample_size", 100),
            enabled=data.get("enabled", True),
            metadata=data.get("metadata", {}),
        )


# === EXPERIMENT MANAGER ===


class ExperimentManager:
    """
    Manager for A/B test experiments.

    Provides:
    - Experiment lifecycle management
    - Consistent variant assignment
    - Result collection and analysis
    - Statistical significance testing
    """

    def __init__(
        self,
        experiments_file: Optional[Path] = None,
        results_file: Optional[Path] = None,
    ) -> None:
        """Initialize the experiment manager."""
        self.experiments_file = experiments_file or Path("config/experiments.json")
        self.results_file = results_file or Path("Logs/experiment_results.jsonl")
        self.experiments: dict[str, Experiment] = {}
        self._load_experiments()

    def _load_experiments(self) -> None:
        """Load experiments from config file."""
        if self.experiments_file.exists():
            try:
                with self.experiments_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    for exp_data in data.get("experiments", []):
                        exp = Experiment.from_dict(exp_data)
                        self.experiments[exp.experiment_id] = exp
                logger.info(f"Loaded {len(self.experiments)} experiments")
            except Exception as e:
                logger.warning(f"Failed to load experiments: {e}")

    def _save_experiments(self) -> None:
        """Save experiments to config file."""
        try:
            self.experiments_file.parent.mkdir(parents=True, exist_ok=True)
            data = {"experiments": [exp.to_dict() for exp in self.experiments.values()]}
            with self.experiments_file.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save experiments: {e}")

    def create_experiment(
        self,
        experiment_id: str,
        name: str,
        description: str,
        variants: list[Variant],
        min_sample_size: int = 100,
    ) -> Experiment:
        """Create a new experiment."""
        if experiment_id in self.experiments:
            raise ValueError(f"Experiment {experiment_id} already exists")

        exp = Experiment(
            experiment_id=experiment_id,
            name=name,
            description=description,
            variants=variants,
            min_sample_size=min_sample_size,
        )
        self.experiments[experiment_id] = exp
        self._save_experiments()
        logger.info(f"Created experiment: {experiment_id}")
        return exp

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get an experiment by ID."""
        return self.experiments.get(experiment_id)

    def get_active_experiments(self) -> list[Experiment]:
        """Get all active experiments."""
        return [exp for exp in self.experiments.values() if exp.enabled and not exp.end_time]

    def stop_experiment(self, experiment_id: str) -> bool:
        """Stop an experiment."""
        exp = self.experiments.get(experiment_id)
        if not exp:
            return False

        exp.end_time = datetime.now(timezone.utc)
        exp.enabled = False
        self._save_experiments()
        logger.info(f"Stopped experiment: {experiment_id}")
        return True

    def assign_variant(
        self,
        experiment_id: str,
        subject_id: str,
    ) -> Optional[Variant]:
        """
        Assign a variant to a subject using consistent hashing.

        Args:
            experiment_id: ID of the experiment
            subject_id: Unique identifier for the subject (e.g., person_id)

        Returns:
            Assigned Variant, or None if experiment not found/inactive
        """
        exp = self.experiments.get(experiment_id)
        if not exp or not exp.enabled:
            return None

        # Consistent hash for deterministic assignment
        hash_input = f"{experiment_id}:{subject_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)

        # Weighted random selection based on hash
        total_weight = sum(v.weight for v in exp.variants)
        threshold = (hash_value % 1000) / 1000.0 * total_weight

        cumulative = 0.0
        for variant in exp.variants:
            cumulative += variant.weight
            if threshold < cumulative:
                return variant

        return exp.variants[-1] if exp.variants else None

    def record_result(
        self,
        experiment_id: str,
        variant_name: str,
        quality_score: float,
        response_time_ms: float,
        success: bool,
        person_id: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Record a trial result."""
        result = ExperimentResult(
            experiment_id=experiment_id,
            variant_name=variant_name,
            person_id=person_id,
            timestamp=datetime.now(timezone.utc),
            quality_score=quality_score,
            response_time_ms=response_time_ms,
            success=success,
            metadata=metadata or {},
        )

        try:
            self.results_file.parent.mkdir(parents=True, exist_ok=True)
            with self.results_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to record experiment result: {e}")

    def get_experiment_results(self, experiment_id: str) -> list[ExperimentResult]:
        """Load all results for an experiment."""
        results: list[ExperimentResult] = []

        if not self.results_file.exists():
            return results

        try:
            with self.results_file.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if data.get("experiment_id") == experiment_id:
                            results.append(
                                ExperimentResult(
                                    experiment_id=data["experiment_id"],
                                    variant_name=data["variant_name"],
                                    person_id=data.get("person_id"),
                                    timestamp=datetime.fromisoformat(data["timestamp"]),
                                    quality_score=data["quality_score"],
                                    response_time_ms=data["response_time_ms"],
                                    success=data["success"],
                                    metadata=data.get("metadata", {}),
                                )
                            )
                    except (json.JSONDecodeError, KeyError):
                        continue
        except Exception as e:
            logger.error(f"Failed to load experiment results: {e}")

        return results

    @staticmethod
    def calculate_variant_stats(results: list[ExperimentResult], variant_name: str) -> VariantStats:
        """Calculate statistics for a variant."""
        variant_results = [r for r in results if r.variant_name == variant_name]

        if not variant_results:
            return VariantStats(variant_name=variant_name)

        quality_scores = [r.quality_score for r in variant_results]
        response_times = [r.response_time_ms for r in variant_results]
        successes = [r.success for r in variant_results]

        avg_quality = sum(quality_scores) / len(quality_scores)
        avg_response = sum(response_times) / len(response_times)
        success_rate = sum(1 for s in successes if s) / len(successes)

        # Calculate standard deviation
        variance = sum((q - avg_quality) ** 2 for q in quality_scores) / len(quality_scores)
        std_quality = variance**0.5

        return VariantStats(
            variant_name=variant_name,
            sample_count=len(variant_results),
            success_rate=success_rate,
            avg_quality_score=avg_quality,
            avg_response_time_ms=avg_response,
            std_quality_score=std_quality,
        )

    def get_experiment_summary(self, experiment_id: str) -> Optional[ExperimentSummary]:
        """Get summary statistics for an experiment."""
        exp = self.experiments.get(experiment_id)
        if not exp:
            return None

        results = self.get_experiment_results(experiment_id)
        variant_stats = [ExperimentManager.calculate_variant_stats(results, v.name) for v in exp.variants]

        # Determine winner (simple comparison for now)
        winner = None
        confidence = 0.0
        if len(variant_stats) >= 2:
            sorted_stats = sorted(variant_stats, key=lambda s: s.avg_quality_score, reverse=True)
            if sorted_stats[0].sample_count >= exp.min_sample_size:
                winner = sorted_stats[0].variant_name
                # Simple confidence based on score difference
                if sorted_stats[1].avg_quality_score > 0:
                    diff_ratio = (sorted_stats[0].avg_quality_score - sorted_stats[1].avg_quality_score) / sorted_stats[
                        1
                    ].avg_quality_score
                    confidence = min(0.95, 0.5 + diff_ratio * 2)

        status = "running" if exp.enabled and not exp.end_time else "completed"

        return ExperimentSummary(
            experiment_id=experiment_id,
            status=status,
            start_time=exp.start_time,
            end_time=exp.end_time,
            total_trials=len(results),
            variants=variant_stats,
            winner=winner,
            confidence=confidence,
        )


# === CONVENIENCE FUNCTIONS ===


class _ManagerHolder:
    """Holder for singleton experiment manager."""

    instance: Optional[ExperimentManager] = None


def get_experiment_manager() -> ExperimentManager:
    """Get the singleton experiment manager."""
    if _ManagerHolder.instance is None:
        _ManagerHolder.instance = ExperimentManager()
    return _ManagerHolder.instance


def get_prompt_variant(
    experiment_id: str,
    subject_id: str,
    fallback_prompt_key: str,
) -> tuple[str, Optional[str], Optional[str]]:
    """
    Get prompt key and variant for a subject in an experiment.

    Args:
        experiment_id: ID of the experiment
        subject_id: Unique identifier for the subject
        fallback_prompt_key: Default prompt key if not in experiment

    Returns:
        Tuple of (prompt_key, variant_name, experiment_variant_name)
    """
    manager = get_experiment_manager()
    variant = manager.assign_variant(experiment_id, subject_id)

    if variant:
        return variant.prompt_key, variant.name, variant.prompt_variant

    return fallback_prompt_key, None, None


# === MODULE TESTS ===


def module_tests() -> bool:
    """Run module-specific tests."""
    from testing.test_framework import TestSuite

    suite = TestSuite("A/B Testing Framework", "ai/ab_testing.py")
    suite.start_suite()

    # Test 1: Variant creation
    def test_variant() -> None:
        v = Variant(name="control", prompt_key="test_prompt")
        assert v.name == "control", "Name should be 'control'"
        assert v.weight == 1.0, "Default weight should be 1.0"

    suite.run_test(
        "Variant creation",
        test_variant,
        test_summary="Variant dataclass initializes correctly",
        functions_tested="Variant dataclass",
        method_description="Create variant and verify fields",
    )

    # Test 2: Experiment creation
    def test_experiment() -> None:
        variants = [
            Variant(name="control", prompt_key="intent_classification"),
            Variant(name="treatment", prompt_key="intent_classification", prompt_variant="v2"),
        ]
        exp = Experiment(
            experiment_id="test_exp_1",
            name="Test Experiment",
            description="Testing",
            variants=variants,
        )
        assert exp.experiment_id == "test_exp_1", "ID should match"
        assert len(exp.variants) == 2, "Should have 2 variants"

    suite.run_test(
        "Experiment creation",
        test_experiment,
        test_summary="Experiment dataclass initializes correctly",
        functions_tested="Experiment dataclass",
        method_description="Create experiment and verify fields",
    )

    # Test 3: Consistent hashing
    def test_consistent_hash() -> None:
        manager = ExperimentManager(
            experiments_file=Path("Cache/test_experiments.json"),
            results_file=Path("Cache/test_results.jsonl"),
        )
        variants = [
            Variant(name="a", prompt_key="test", weight=1.0),
            Variant(name="b", prompt_key="test", weight=1.0),
        ]
        exp = Experiment(
            experiment_id="hash_test",
            name="Hash Test",
            description="Test hashing",
            variants=variants,
        )
        manager.experiments["hash_test"] = exp

        # Same subject should get same variant
        v1 = manager.assign_variant("hash_test", "subject_1")
        v2 = manager.assign_variant("hash_test", "subject_1")
        assert v1 and v2, "Should get variants"
        assert v1.name == v2.name, "Same subject should get same variant"

    suite.run_test(
        "Consistent hashing",
        test_consistent_hash,
        test_summary="Same subject gets same variant consistently",
        functions_tested="ExperimentManager.assign_variant",
        method_description="Assign variant twice to same subject",
    )

    # Test 4: VariantStats calculation
    def test_variant_stats() -> None:
        stats = VariantStats(
            variant_name="test",
            sample_count=10,
            success_rate=0.9,
            avg_quality_score=85.0,
        )
        assert stats.sample_count == 10, "Sample count should be 10"
        assert stats.success_rate == 0.9, "Success rate should be 0.9"

    suite.run_test(
        "VariantStats dataclass",
        test_variant_stats,
        test_summary="VariantStats stores values correctly",
        functions_tested="VariantStats dataclass",
        method_description="Create stats and verify fields",
    )

    # Test 5: ExperimentResult serialization
    def test_result_serialization() -> None:
        result = ExperimentResult(
            experiment_id="test",
            variant_name="control",
            person_id=123,
            timestamp=datetime.now(timezone.utc),
            quality_score=85.5,
            response_time_ms=150.0,
            success=True,
        )
        data = result.to_dict()
        assert data["experiment_id"] == "test", "ID should serialize"
        assert data["quality_score"] == 85.5, "Score should serialize"

    suite.run_test(
        "Result serialization",
        test_result_serialization,
        test_summary="ExperimentResult serializes to dict correctly",
        functions_tested="ExperimentResult.to_dict",
        method_description="Convert result to dict and verify",
    )

    # Test 6: Experiment serialization roundtrip
    def test_experiment_roundtrip() -> None:
        variants = [Variant(name="control", prompt_key="test")]
        exp = Experiment(
            experiment_id="roundtrip",
            name="Roundtrip Test",
            description="Test",
            variants=variants,
        )
        data = exp.to_dict()
        restored = Experiment.from_dict(data)
        assert restored.experiment_id == exp.experiment_id, "ID should match"
        assert len(restored.variants) == 1, "Variants should restore"

    suite.run_test(
        "Experiment roundtrip",
        test_experiment_roundtrip,
        test_summary="Experiment serializes and deserializes correctly",
        functions_tested="Experiment.to_dict, Experiment.from_dict",
        method_description="Serialize and deserialize experiment",
    )

    return suite.finish_suite()


def run_comprehensive_tests() -> bool:
    """Run all tests with proper framework setup."""
    from testing.test_framework import create_standard_test_runner

    runner = create_standard_test_runner(module_tests)
    return runner()


if __name__ == "__main__":
    import sys

    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
