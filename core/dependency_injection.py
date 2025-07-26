#!/usr/bin/env python3

"""
Dependency Injection Framework.

This module provides a comprehensive dependency injection system to:
- Resolve circular import issues
- Manage component dependencies
- Support singleton and factory patterns
- Enable clean testing with mocks
- Provide configuration-driven component instantiation
"""

# === CORE INFRASTRUCTURE ===
import sys
import os

# Add parent directory to path for standard_imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from standard_imports import setup_module

logger = setup_module(globals(), __name__)

# === PHASE 4.1: ENHANCED ERROR HANDLING ===
from error_handling import (
    retry_on_failure,
    circuit_breaker,
    timeout_protection,
    graceful_degradation,
    error_context,
    AncestryException,
    RetryableError,
    NetworkTimeoutError,
    AuthenticationExpiredError,
    APIRateLimitError,
    ErrorContext,
)

# === STANDARD LIBRARY IMPORTS ===
import logging
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, Type, TypeVar, Optional, Callable, Union, List
from functools import wraps
import inspect
import sys
import os
import unittest
from unittest.mock import MagicMock

T = TypeVar("T")


class DIContainer:
    """
    Dependency Injection Container.

    Provides registration and resolution of dependencies with support for:
    - Singleton instances
    - Factory functions
    - Interface-to-implementation mapping
    - Lifecycle management
    - Thread safety
    """

    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._singletons: Dict[str, Any] = {}
        self._interfaces: Dict[Type, Type] = {}
        self._lock = threading.RLock()
        self._initialization_order: List[str] = []

    def register_singleton(
        self,
        interface: Type[T],
        implementation: Union[Type[T], T],
        name: Optional[str] = None,
    ) -> None:
        """
        Register a singleton service.

        Args:
            interface: Service interface or type
            implementation: Implementation class or instance
            name: Optional service name
        """
        with self._lock:
            service_name = name or self._get_service_name(interface)

            if inspect.isclass(implementation):
                # Store class for lazy instantiation
                self._services[service_name] = implementation
                self._interfaces[interface] = implementation
                logger.debug(f"Registered singleton class: {service_name}")
            else:
                # Store instance directly
                self._singletons[service_name] = implementation
                self._interfaces[interface] = type(implementation)
                logger.debug(f"Registered singleton instance: {service_name}")

    def register_transient(
        self, interface: Type[T], implementation: Type[T], name: Optional[str] = None
    ) -> None:
        """
        Register a transient service (new instance each time).

        Args:
            interface: Service interface or type
            implementation: Implementation class
            name: Optional service name
        """
        with self._lock:
            service_name = name or self._get_service_name(interface)

            def factory():
                return self._create_instance(implementation)

            self._factories[service_name] = factory
            self._interfaces[interface] = implementation
            logger.debug(f"Registered transient service: {service_name}")

    def register_factory(
        self, interface: Type[T], factory: Callable[[], T], name: Optional[str] = None
    ) -> None:
        """
        Register a factory function.

        Args:
            interface: Service interface or type
            factory: Factory function that creates the service
            name: Optional service name
        """
        with self._lock:
            service_name = name or self._get_service_name(interface)
            self._factories[service_name] = factory
            logger.debug(f"Registered factory: {service_name}")

    def register_instance(
        self, interface: Type[T], instance: T, name: Optional[str] = None
    ) -> None:
        """
        Register a specific instance.

        Args:
            interface: Service interface or type
            instance: Service instance
            name: Optional service name
        """
        with self._lock:
            service_name = name or self._get_service_name(interface)
            self._singletons[service_name] = instance
            self._interfaces[interface] = type(instance)
            logger.debug(f"Registered instance: {service_name}")

    def resolve(self, interface: Type[T], name: Optional[str] = None) -> T:
        """
        Resolve a service instance.

        Args:
            interface: Service interface or type
            name: Optional service name

        Returns:
            Service instance

        Raises:
            DIResolutionError: If service cannot be resolved
        """
        with self._lock:
            service_name = name or self._get_service_name(interface)

            # Check if already instantiated as singleton
            if service_name in self._singletons:
                return self._singletons[service_name]

            # Check for factory
            if service_name in self._factories:
                instance = self._factories[service_name]()
                logger.debug(f"Created instance via factory: {service_name}")
                return instance

            # Check for registered class (singleton)
            if service_name in self._services:
                instance = self._create_instance(self._services[service_name])
                self._singletons[service_name] = instance
                logger.debug(f"Created singleton instance: {service_name}")
                return instance

            # Try to resolve by interface mapping
            if interface in self._interfaces:
                implementation = self._interfaces[interface]
                instance = self._create_instance(implementation)
                self._singletons[service_name] = instance
                logger.debug(f"Created instance via interface mapping: {service_name}")
                return instance

            raise DIResolutionError(
                f"Cannot resolve service: {service_name} ({interface})"
            )

    def is_registered(self, interface: Type, name: Optional[str] = None) -> bool:
        """
        Check if a service is registered.

        Args:
            interface: Service interface or type
            name: Optional service name

        Returns:
            True if service is registered
        """
        with self._lock:
            service_name = name or self._get_service_name(interface)
            return (
                service_name in self._services
                or service_name in self._factories
                or service_name in self._singletons
                or interface in self._interfaces
            )

    def clear(self) -> None:
        """Clear all registered services."""
        with self._lock:
            self._services.clear()
            self._factories.clear()
            self._singletons.clear()
            self._interfaces.clear()
            self._initialization_order.clear()
            logger.debug("DI container cleared")

    def get_registration_info(self) -> Dict[str, Any]:
        """
        Get information about registered services.

        Returns:
            Dictionary with registration information
        """
        with self._lock:
            return {
                "singleton_classes": list(self._services.keys()),
                "singleton_instances": list(self._singletons.keys()),
                "factories": list(self._factories.keys()),
                "interface_mappings": {
                    str(interface): str(implementation)
                    for interface, implementation in self._interfaces.items()
                },
                "total_registrations": len(self._services)
                + len(self._factories)
                + len(self._singletons),
            }

    def _get_service_name(self, service_type: Type) -> str:
        """Get service name from type."""
        return f"{service_type.__module__}.{service_type.__name__}"

    def _create_instance(self, implementation: Type) -> Any:
        """
        Create an instance with dependency injection.

        Args:
            implementation: Implementation class

        Returns:
            Created instance
        """
        try:
            # Get constructor signature
            signature = inspect.signature(implementation.__init__)
            parameters = signature.parameters

            # Skip 'self' parameter
            constructor_params = {
                name: param for name, param in parameters.items() if name != "self"
            }

            # Resolve dependencies
            kwargs = {}
            for param_name, param in constructor_params.items():
                if param.annotation != inspect.Parameter.empty:
                    try:
                        # Try to resolve the parameter type
                        dependency = self.resolve(param.annotation)
                        kwargs[param_name] = dependency
                    except DIResolutionError:
                        # If dependency cannot be resolved and parameter has no default,
                        # let the constructor handle it
                        if param.default == inspect.Parameter.empty:
                            logger.warning(
                                f"Cannot resolve dependency {param_name}: {param.annotation} "
                                f"for {implementation.__name__}"
                            )

            # Create instance
            instance = implementation(**kwargs)
            logger.debug(
                f"Created instance: {implementation.__name__} with dependencies: {list(kwargs.keys())}"
            )
            return instance

        except Exception as e:
            logger.error(f"Failed to create instance of {implementation}: {e}")
            # Fallback to parameterless constructor
            try:
                return implementation()
            except Exception as fallback_error:
                logger.error(f"Fallback creation also failed: {fallback_error}")
                raise DIResolutionError(
                    f"Cannot create instance of {implementation}: {e}"
                )


class DIResolutionError(Exception):
    """Dependency injection resolution error."""

    pass


class Injectable:
    """
    Base class for injectable services.

    Provides automatic registration and dependency resolution.
    """

    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses."""
        super().__init_subclass__(**kwargs)
        # Note: Actual registration happens in the container setup
        logger.debug(f"Injectable subclass defined: {cls.__name__}")


def inject(service_type: Type[T], name: Optional[str] = None) -> Callable:
    """
    Decorator for dependency injection.

    Args:
        service_type: Type of service to inject
        name: Optional service name

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get container from global registry
            container = get_container()
            service_instance = container.resolve(service_type, name)

            # Add service to kwargs
            service_param_name = name or service_type.__name__.lower()
            kwargs[service_param_name] = service_instance

            return func(*args, **kwargs)

        return wrapper

    return decorator


class ServiceRegistry:
    """
    Global service registry for managing DI container instances.
    """

    _containers: Dict[str, DIContainer] = {}
    _default_container: Optional[DIContainer] = None
    _lock = threading.RLock()

    @classmethod
    def get_container(cls, name: str = "default") -> DIContainer:
        """
        Get or create a DI container.

        Args:
            name: Container name

        Returns:
            DI container instance
        """
        with cls._lock:
            if name not in cls._containers:
                cls._containers[name] = DIContainer()
                logger.debug(f"Created DI container: {name}")

            container = cls._containers[name]

            if name == "default" and cls._default_container is None:
                cls._default_container = container

            return container

    @classmethod
    def clear_container(cls, name: str = "default") -> None:
        """
        Clear a specific container.

        Args:
            name: Container name
        """
        with cls._lock:
            if name in cls._containers:
                cls._containers[name].clear()
                if name == "default":
                    cls._default_container = None
                logger.debug(f"Cleared DI container: {name}")

    @classmethod
    def clear_all_containers(cls) -> None:
        """Clear all containers."""
        with cls._lock:
            for container in cls._containers.values():
                container.clear()
            cls._containers.clear()
            cls._default_container = None
            logger.debug("Cleared all DI containers")


def get_container(name: str = "default") -> DIContainer:
    """
    Get the DI container.

    Args:
        name: Container name

    Returns:
        DI container instance
    """
    return ServiceRegistry.get_container(name)


def configure_dependencies():
    """
    Configure application dependencies.

    This function sets up the dependency injection container
    with all the application services.
    """
    container = get_container()

    # Clear existing registrations
    container.clear()

    try:
        # Register core services
        from core.database_manager import DatabaseManager
        from core.browser_manager import BrowserManager
        from core.api_manager import APIManager
        from core.session_validator import SessionValidator
        from core.session_manager import SessionManager

        # Register managers as singletons
        container.register_singleton(DatabaseManager, DatabaseManager)
        container.register_singleton(BrowserManager, BrowserManager)
        container.register_singleton(APIManager, APIManager)
        container.register_singleton(SessionValidator, SessionValidator)
        container.register_singleton(SessionManager, SessionManager)

        logger.info("Core services registered in DI container")

    except ImportError as e:
        logger.warning(f"Could not register core services: {e}")

    try:
        # Register configuration services
        from config.config_manager import ConfigManager

        container.register_singleton(ConfigManager, ConfigManager)

        logger.info("Configuration services registered in DI container")

    except ImportError as e:
        logger.warning(f"Could not register configuration services: {e}")

    try:
        # Register security services
        from security_manager import SecurityManager

        container.register_singleton(SecurityManager, SecurityManager)

        logger.info("Security services registered in DI container")

    except ImportError as e:
        logger.warning(f"Could not register security services: {e}")

    logger.info("Dependency injection configuration completed")


def get_service(service_type: Type[T], name: Optional[str] = None) -> T:
    """
    Convenience function to get a service from the default container.

    Args:
        service_type: Service type
        name: Optional service name

    Returns:
        Service instance
    """
    container = get_container()
    return container.resolve(service_type, name)


# Context manager for DI scope
class DIScope:
    """
    Context manager for DI container scope.

    Allows temporary service registrations within a specific scope.
    """

    def __init__(self, container_name: str = "default"):
        self.container_name = container_name
        self._original_registrations = None

    def __enter__(self) -> DIContainer:
        container = get_container(self.container_name)

        # Save current state
        self._original_registrations = {
            "services": dict(container._services),
            "factories": dict(container._factories),
            "singletons": dict(container._singletons),
            "interfaces": dict(container._interfaces),
        }

        return container

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._original_registrations:
            container = get_container(self.container_name)

            # Restore original state
            container._services = self._original_registrations["services"]
            container._factories = self._original_registrations["factories"]
            container._singletons = self._original_registrations["singletons"]
            container._interfaces = self._original_registrations["interfaces"]


class TestDIContainer(unittest.TestCase):
    def setUp(self):
        self.container = DIContainer()

    def test_register_singleton(self):
        class ServiceA:
            pass

        self.container.register_singleton(ServiceA, ServiceA)
        instance1 = self.container.resolve(ServiceA)
        instance2 = self.container.resolve(ServiceA)
        self.assertIs(instance1, instance2)

    def test_register_transient(self):
        class ServiceB:
            pass

        self.container.register_transient(ServiceB, ServiceB)
        instance1 = self.container.resolve(ServiceB)
        instance2 = self.container.resolve(ServiceB)
        self.assertIsNot(instance1, instance2)

    def test_register_factory(self):
        class ServiceC:
            def __init__(self, value):
                self.value = value

        def factory():
            return ServiceC("factory_value")

        self.container.register_factory(ServiceC, factory)
        instance = self.container.resolve(ServiceC)
        self.assertEqual(instance.value, "factory_value")

    def test_register_instance(self):
        class ServiceD:
            pass

        instance = ServiceD()
        self.container.register_instance(ServiceD, instance)
        resolved_instance = self.container.resolve(ServiceD)
        self.assertIs(resolved_instance, instance)

    def test_resolve_singleton(self):
        class ServiceE:
            pass

        self.container.register_singleton(ServiceE, ServiceE)
        instance = self.container.resolve(ServiceE)
        self.assertIsInstance(instance, ServiceE)

    def test_resolve_transient(self):
        class ServiceF:
            pass

        self.container.register_transient(ServiceF, ServiceF)
        instance = self.container.resolve(ServiceF)
        self.assertIsInstance(instance, ServiceF)

    def test_resolve_factory(self):
        class ServiceG:
            def __init__(self, value):
                self.value = value

        def factory():
            return ServiceG("factory_value")

        self.container.register_factory(ServiceG, factory)
        instance = self.container.resolve(ServiceG)
        self.assertIsInstance(instance, ServiceG)
        self.assertEqual(instance.value, "factory_value")

    def test_resolve_instance(self):
        class ServiceH:
            pass

        instance = ServiceH()
        self.container.register_instance(ServiceH, instance)
        resolved_instance = self.container.resolve(ServiceH)
        self.assertIsInstance(resolved_instance, ServiceH)
        self.assertIs(resolved_instance, instance)

    def test_is_registered(self):
        class ServiceI:
            pass

        self.assertFalse(self.container.is_registered(ServiceI))
        self.container.register_singleton(ServiceI, ServiceI)
        self.assertTrue(self.container.is_registered(ServiceI))

    def test_clear(self):
        class ServiceJ:
            pass

        self.container.register_singleton(ServiceJ, ServiceJ)
        self.container.clear()
        self.assertFalse(self.container.is_registered(ServiceJ))

    def test_get_registration_info(self):
        class ServiceK:
            pass

        self.container.register_singleton(ServiceK, ServiceK, "service_k")
        info = self.container.get_registration_info()
        self.assertIn("service_k", info["singleton_instances"])

    def test_create_instance(self):
        class ServiceL:
            def __init__(self, value):
                self.value = value

        self.container.register_singleton(ServiceL, ServiceL)
        instance = self.container.resolve(ServiceL)
        self.assertIsInstance(instance, ServiceL)

    def test_di_resolution_error(self):
        class UnregisteredService:
            pass

        with self.assertRaises(DIResolutionError):
            self.container.resolve(UnregisteredService)

    def test_injectable_class(self):
        class InjectableService(Injectable):
            def __init__(self):
                self.injected = True

        service = InjectableService()
        self.assertIsInstance(service, Injectable)
        self.assertTrue(service.injected)

    def test_inject_decorator(self):
        class InjectedService:
            def __init__(self):
                self.value = "injected"

        self.container.register_singleton(InjectedService, InjectedService)

        @inject(InjectedService)
        def test_function(**kwargs):
            service = kwargs.get("injectedservice")
            return service.value if service else "not_injected"

        result = test_function()
        self.assertEqual(result, "injected")

    def test_service_registry(self):
        container1 = ServiceRegistry.get_container("test_container")
        self.assertIsNotNone(container1)

        container2 = ServiceRegistry.get_container("test_container")
        self.assertIs(container1, container2)

        default_container = ServiceRegistry.get_container()
        self.assertIsNotNone(default_container)

        ServiceRegistry.clear_container("test_container")
        ServiceRegistry.clear_all_containers()

    def test_global_container(self):
        container1 = get_container()
        container2 = get_container()

        self.assertIs(container1, container2)

        named_container = get_container("test_container")
        self.assertIsNot(container1, named_container)

    def test_configure_dependencies(self):
        try:
            configure_dependencies()
        except Exception as e:
            self.fail(f"configure_dependencies raised an exception: {e}")

    def test_get_service_convenience(self):
        class ConvenienceService:
            def __init__(self):
                self.convenient = True

        self.container.register_singleton(ConvenienceService, ConvenienceService)

        service = get_service(ConvenienceService)
        self.assertIsInstance(service, ConvenienceService)
        self.assertTrue(service.convenient)

    def test_di_scope(self):
        with DIScope() as container:

            class ScopedService:
                def __init__(self):
                    self.scoped = True

            container.register_singleton(ScopedService, ScopedService)

            service = container.resolve(ScopedService)
            self.assertIsInstance(service, ScopedService)
            self.assertTrue(service.scoped)

    def test_container_state_management(self):
        container = DIContainer()

        class StateService:
            def __init__(self):
                self.state = "initial"

        container.register_singleton(StateService, StateService)

        service = container.resolve(StateService)
        service.state = "modified"

        service2 = container.resolve(StateService)
        self.assertIs(service2, service)
        self.assertEqual(service2.state, "modified")

    def test_type_annotations(self):
        import inspect

        sig = inspect.signature(DIContainer.register_singleton)
        self.assertIn("interface", sig.parameters)
        self.assertIn("implementation", sig.parameters)

        sig = inspect.signature(get_service)
        self.assertIn("service_type", sig.parameters)

    def test_imports_and_availability(self):
        self.assertIsNotNone(DIContainer)
        self.assertIsNotNone(DIResolutionError)
        self.assertIsNotNone(Injectable)
        self.assertIsNotNone(ServiceRegistry)
        self.assertIsNotNone(DIScope)
        self.assertIsNotNone(inject)
        self.assertIsNotNone(get_container)
        self.assertIsNotNone(configure_dependencies)
        self.assertIsNotNone(get_service)

    def test_integration(self):
        container = get_container()

        class ConfigService:
            def __init__(self):
                self.config = {"key": "value"}

        class LogService:
            def __init__(self, config: ConfigService):
                self.config = config
                self.logged = []

        class AppService:
            def __init__(self, log: LogService, config: ConfigService):
                self.log = log
                self.config = config

        container.register_singleton(ConfigService, ConfigService)
        container.register_singleton(LogService, LogService)
        container.register_singleton(AppService, AppService)

        app = container.resolve(AppService)

        self.assertIsInstance(app, AppService)
        self.assertIsInstance(app.log, LogService)
        self.assertIsInstance(app.config, ConfigService)
        self.assertIs(app.log.config, app.config)  # Define all tests


def run_comprehensive_tests():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestDIContainer))
    runner = unittest.TextTestRunner()
    runner.run(suite)


if __name__ == "__main__":
    # Use centralized path management
    import os
    import sys

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        sys.path.insert(0, project_root)
        from core_imports import ensure_imports

        ensure_imports()
    except ImportError:
        # Fallback for testing environment
        sys.path.insert(0, project_root)
    run_comprehensive_tests()
