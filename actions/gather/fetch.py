from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from testing.test_framework import TestSuite, create_standard_test_runner

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GatherFetchPlan:
    """Represents a lightweight plan for fetching a page of matches."""

    page: int
    expected_matches: int
    resume_from_checkpoint: bool = False


class GatherFetchService:
    """Builds placeholder fetch plans until the legacy helpers migrate here."""

    def __init__(self, default_batch_size: int) -> None:
        if default_batch_size <= 0:
            raise ValueError("default_batch_size must be positive")
        self._default_batch_size = default_batch_size

    @property
    def default_batch_size(self) -> int:
        """Expose the configured default batch size."""

        return self._default_batch_size

    def build_plan(self, page: int, matches_on_page: Optional[int]) -> GatherFetchPlan:
        """Create a fetch plan with basic bookkeeping for tests."""

        expected = matches_on_page if matches_on_page and matches_on_page > 0 else self._default_batch_size
        resume_flag = matches_on_page is None
        logger.debug("Prepared fetch plan", extra={"page": page, "expected": expected, "resume": resume_flag})
        return GatherFetchPlan(page=page, expected_matches=expected, resume_from_checkpoint=resume_flag)


# ---------------------------------------------------------------------------
# Module tests
# ---------------------------------------------------------------------------


def _test_fetch_service_defaults_to_batch_size() -> bool:
    service = GatherFetchService(default_batch_size=25)
    plan = service.build_plan(page=3, matches_on_page=None)
    assert plan.page == 3
    assert plan.expected_matches == 25
    assert plan.resume_from_checkpoint is True
    return True


def _test_fetch_service_uses_explicit_match_count() -> bool:
    service = GatherFetchService(default_batch_size=25)
    plan = service.build_plan(page=4, matches_on_page=18)
    assert plan.page == 4
    assert plan.expected_matches == 18
    assert plan.resume_from_checkpoint is False
    return True


def module_tests() -> bool:
    suite = TestSuite("actions.gather.fetch", "actions/gather/fetch.py")
    suite.run_test(
        "Defaults to batch size",
        _test_fetch_service_defaults_to_batch_size,
        "Ensures the fetch plan uses the batch size when resume metadata is absent.",
    )
    suite.run_test(
        "Uses explicit match counts",
        _test_fetch_service_uses_explicit_match_count,
        "Ensures the fetch plan honors known per-page match counts.",
    )
    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)


if __name__ == "__main__":
    success = module_tests()
    raise SystemExit(0 if success else 1)
