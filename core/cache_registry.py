#!/usr/bin/env python3
"""Central registry that coordinates all cache subsystems.

The cache registry exposes a single interface for querying statistics, warming,
and clearing the different cache implementations spread across the project.
It lazily imports the underlying modules to avoid circular dependencies and
keeps light-weight metadata for operations dashboards.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Optional

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from standard_imports import setup_module

logger = setup_module(globals(), __name__)

StatsFn = Callable[[], dict[str, Any]]
ActionFn = Callable[[], Any]


@dataclass
class CacheComponent:
    """Metadata describing a cache subsystem."""

    name: str
    kind: str
    stats_fn: StatsFn
    clear_fn: Optional[ActionFn] = None
    warm_fn: Optional[ActionFn] = None
    health_fn: Optional[StatsFn] = None

    def safe_stats(self) -> dict[str, Any]:
        try:
            stats = self.stats_fn()
            stats.setdefault("name", self.name)
            stats.setdefault("kind", self.kind)
            return stats
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Cache stats failed for %s: %s", self.name, exc)
            return {"name": self.name, "kind": self.kind, "error": str(exc)}

    def safe_health(self) -> Optional[dict[str, Any]]:
        if not self.health_fn:
            return None
        try:
            details = self.health_fn()
            details.setdefault("name", self.name)
            return details
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Cache health failed for %s: %s", self.name, exc)
            return {"name": self.name, "error": str(exc)}

    def safe_clear(self) -> bool:
        if not self.clear_fn:
            return False
        try:
            self.clear_fn()
            return True
        except Exception:  # pragma: no cover - defensive logging
            logger.warning("Cache clear failed for %s", self.name, exc_info=True)
            return False

    def safe_warm(self) -> bool:
        if not self.warm_fn:
            return False
        try:
            self.warm_fn()
            return True
        except Exception:  # pragma: no cover - defensive logging
            logger.warning("Cache warm failed for %s", self.name, exc_info=True)
            return False


class CacheRegistry:
    """Registry responsible for orchestrating cache components."""

    def __init__(self) -> None:
        self._components: dict[str, CacheComponent] = {}
        self._lock = RLock()
        self._register_defaults()

    # --- public API -----------------------------------------------------

    def register(self, component: CacheComponent) -> None:
        with self._lock:
            self._components[component.name] = component

    def component_names(self) -> list[str]:
        with self._lock:
            return sorted(self._components.keys())

    def component_stats(self, name: str) -> dict[str, Any]:
        component = self._components.get(name)
        return component.safe_stats() if component else {}

    def summary(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            summary = {name: comp.safe_stats() for name, comp in self._components.items()}
            for name, comp in self._components.items():
                health = comp.safe_health()
                if health:
                    summary[name]["health"] = health
            summary["registry"] = {
                "components": len(self._components),
                "names": self.component_names(),
            }
            return summary

    def clear(self, name: Optional[str] = None) -> dict[str, bool]:
        results: dict[str, bool] = {}
        with self._lock:
            for comp_name, component in self._components.items():
                if name is None or comp_name == name:
                    results[comp_name] = component.safe_clear()
        return results

    def warm(self, name: Optional[str] = None) -> dict[str, bool]:
        results: dict[str, bool] = {}
        with self._lock:
            for comp_name, component in self._components.items():
                if name is None or comp_name == name:
                    results[comp_name] = component.safe_warm()
        return results

    # --- registration helpers ------------------------------------------

    def _register_defaults(self) -> None:
        self.register(
            CacheComponent(
                name="disk_cache",
                kind="disk",
                stats_fn=self._lazy_call("cache", "get_cache_stats"),
                clear_fn=self._lazy_call("cache", "clear_cache"),
            )
        )

        self.register(
            CacheComponent(
                name="unified_cache",
                kind="memory",
                stats_fn=self._unified_cache_stats,
                clear_fn=self._unified_cache_clear,
            )
        )

        self.register(
            CacheComponent(
                name="session_cache",
                kind="session",
                stats_fn=self._lazy_call("core.session_cache", "get_session_cache_stats"),
                clear_fn=self._lazy_call("core.session_cache", "clear_session_cache"),
                warm_fn=self._lazy_call("core.session_cache", "warm_session_cache"),
            )
        )

        self.register(
            CacheComponent(
                name="system_cache",
                kind="system",
                stats_fn=self._lazy_call("core.system_cache", "get_system_cache_stats"),
                clear_fn=self._lazy_call("core.system_cache", "clear_system_caches"),
                warm_fn=self._lazy_call("core.system_cache", "warm_system_caches"),
            )
        )

        self.register(
            CacheComponent(
                name="gedcom_cache",
                kind="gedcom",
                stats_fn=self._lazy_call("gedcom_cache", "get_gedcom_cache_stats"),
                clear_fn=self._lazy_call("gedcom_cache", "clear_memory_cache"),
                warm_fn=self._lazy_call("gedcom_cache", "warm_gedcom_cache"),
                health_fn=self._lazy_call("gedcom_cache", "get_gedcom_cache_health"),
            )
        )

        self.register(
            CacheComponent(
                name="tree_stats_cache",
                kind="database",
                stats_fn=self._lazy_call("tree_stats_utils", "get_tree_stats_cache_stats"),
                clear_fn=self._lazy_call("tree_stats_utils", "clear_tree_stats_cache"),
                warm_fn=self._lazy_call("tree_stats_utils", "warm_tree_stats_cache"),
            )
        )

        self.register(
            CacheComponent(
                name="performance_cache",
                kind="performance",
                stats_fn=self._lazy_call("performance_cache", "get_performance_cache_stats"),
                clear_fn=self._lazy_call("performance_cache", "clear_performance_cache"),
                warm_fn=self._lazy_call("performance_cache", "warm_performance_cache"),
            )
        )

    @staticmethod
    def _lazy_call(module_path: str, attr: str) -> ActionFn:
        def _callable(*args: Any, **kwargs: Any) -> Any:
            module = import_module(module_path)
            target = getattr(module, attr)
            return target(*args, **kwargs)

        return _callable

    @staticmethod
    def _unified_cache_stats() -> dict[str, Any]:
        cache_module = import_module("core.unified_cache_manager")
        unified_cache = cache_module.get_unified_cache()
        return unified_cache.get_stats()

    @staticmethod
    def _unified_cache_clear() -> None:
        cache_module = import_module("core.unified_cache_manager")
        unified_cache = cache_module.get_unified_cache()
        unified_cache.clear()


_registry: Optional[CacheRegistry] = None


def get_cache_registry() -> CacheRegistry:
    """Return the singleton cache registry instance."""
    global _registry  # noqa: PLW0603
    if _registry is None:
        _registry = CacheRegistry()
        logger.debug("Cache registry initialized with %d components", len(_registry.component_names()))
    return _registry


__all__ = ["CacheComponent", "CacheRegistry", "get_cache_registry"]


# === Module Tests ===


def _build_stub_registry() -> tuple[CacheRegistry, dict[str, int]]:
    """Create a CacheRegistry populated with a single stub component for tests."""

    registry = CacheRegistry()
    registry._components = {}
    call_counts: dict[str, int] = {"stats": 0, "clear": 0, "warm": 0, "health": 0}

    def stats_fn() -> dict[str, Any]:
        call_counts["stats"] += 1
        return {"hits": 1}

    def clear_fn() -> None:
        call_counts["clear"] += 1

    def warm_fn() -> None:
        call_counts["warm"] += 1

    def health_fn() -> dict[str, Any]:
        call_counts["health"] += 1
        return {"overall_score": 97.5}

    registry.register(
        CacheComponent(
            name="stub_cache",
            kind="memory",
            stats_fn=stats_fn,
            clear_fn=clear_fn,
            warm_fn=warm_fn,
            health_fn=health_fn,
        )
    )
    return registry, call_counts


def _test_registry_summary_includes_health() -> None:
    registry, call_counts = _build_stub_registry()
    summary = registry.summary()

    assert "stub_cache" in summary, "Summary should include registered component"
    component_stats = summary["stub_cache"]
    assert component_stats["kind"] == "memory"
    assert component_stats["health"]["overall_score"] == 97.5
    assert summary["registry"]["components"] == 1
    assert call_counts["stats"] == 1, "Stats function should be invoked once"
    assert call_counts["health"] == 1, "Health function should be invoked once"


def _test_registry_clear_and_warm_routing() -> None:
    registry, call_counts = _build_stub_registry()
    clear_result = registry.clear()
    warm_result = registry.warm()

    assert clear_result.get("stub_cache") is True, "Clear should succeed for stub cache"
    assert warm_result.get("stub_cache") is True, "Warm should succeed for stub cache"
    assert call_counts["clear"] == 1, "Clear function should be called exactly once"
    assert call_counts["warm"] == 1, "Warm function should be called exactly once"


def _test_registry_component_names_sorted() -> None:
    registry, _ = _build_stub_registry()
    registry.register(
        CacheComponent(
            name="alpha_cache",
            kind="disk",
            stats_fn=lambda: {},
        )
    )
    names = registry.component_names()
    assert names == sorted(names), "Component names should be returned in sorted order"
    assert "alpha_cache" in names and "stub_cache" in names


def cache_registry_module_tests() -> bool:
    """Test suite for cache registry orchestration helpers."""

    from test_framework import TestSuite, suppress_logging

    with suppress_logging():
        suite = TestSuite("Cache Registry", __name__)
        suite.start_suite()
        suite.run_test(
            "Summary includes health",
            _test_registry_summary_includes_health,
            "CacheRegistry.summary aggregates stats and health information",
            "summary",
            "Ensure summary contains component stats and registry metadata",
        )
        suite.run_test(
            "Clear and warm routing",
            _test_registry_clear_and_warm_routing,
            "CacheRegistry proxies clear/warm requests to components",
            "clear/warm",
            "Invoke registry clear/warm and verify underlying functions run",
        )
        suite.run_test(
            "Component names sorted",
            _test_registry_component_names_sorted,
            "component_names returns sorted list",
            "component_names",
            "Verify alphabetical ordering of registry component names",
        )
        return suite.finish_suite()


from test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(cache_registry_module_tests)


if __name__ == "__main__":
    import sys

    sys.exit(0 if run_comprehensive_tests() else 1)
