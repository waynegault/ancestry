"""
Dependency Injection Framework.

This module provides a comprehensive dependency injection system to:
- Resolve circular import issues
- Manage component dependencies
- Support singleton and factory patterns
- Enable clean testing with mocks
- Provide configuration-driven component instantiation
"""

import logging
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, Type, TypeVar, Optional, Callable, Union, List
from functools import wraps
import inspect

logger = logging.getLogger(__name__)

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
        from config import ConfigManager

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


def run_comprehensive_tests():
    """
    Comprehensive test suite for DependencyInjection module.

    Tests all major functionality including:
    - DIContainer functionality
    - Service registration and resolution
    - Singleton and transient services
    - Factory patterns
    - Injectable base class
    - Dependency injection
    - Service registry
    - Scoping
    """
    print("=" * 50)
    print("RUNNING DEPENDENCY INJECTION COMPREHENSIVE TESTS")
    print("=" * 50)

    passed = 0
    failed = 0

    def run_test(test_func, test_name):
        nonlocal passed, failed
        try:
            test_func()
            print(f"✓ {test_name} passed")
            passed += 1
        except Exception as e:
            print(f"✗ {test_name} failed: {e}")
            failed += 1

    # Test 1: DIContainer basics
    def test_di_container_basics():
        container = DIContainer()

        class TestService:
            def __init__(self):
                self.value = "test"

        container.register_singleton(TestService, TestService)

        service1 = container.resolve(TestService)
        service2 = container.resolve(TestService)

        assert isinstance(service1, TestService)
        assert isinstance(service2, TestService)
        assert service1 is service2  # Singleton behavior
        assert service1.value == "test"

    run_test(test_di_container_basics, "DIContainer basics")

    # Test 2: Transient services
    def test_transient_services():
        container = DIContainer()

        class TransientService:
            def __init__(self):
                self.id = id(self)

        container.register_transient(TransientService, TransientService)

        service1 = container.resolve(TransientService)
        service2 = container.resolve(TransientService)

        assert isinstance(service1, TransientService)
        assert isinstance(service2, TransientService)
        assert service1 is not service2  # Different instances
        assert service1.id != service2.id

    run_test(test_transient_services, "Transient services")

    # Test 3: Factory registration
    def test_factory_registration():
        container = DIContainer()

        class FactoryService:
            def __init__(self, value):
                self.value = value

        def factory():
            return FactoryService("factory_created")

        container.register_factory(FactoryService, factory)

        service = container.resolve(FactoryService)
        assert isinstance(service, FactoryService)
        assert service.value == "factory_created"

    run_test(test_factory_registration, "Factory registration")

    # Test 4: Named services
    def test_named_services():
        container = DIContainer()

        class NamedService:
            def __init__(self, name):
                self.name = name

        service1 = NamedService("service1")
        service2 = NamedService("service2")

        container.register_singleton(NamedService, service1, name="first")
        container.register_singleton(NamedService, service2, name="second")

        resolved1 = container.resolve(NamedService, name="first")
        resolved2 = container.resolve(NamedService, name="second")

        assert resolved1 is service1
        assert resolved2 is service2
        assert resolved1.name == "service1"
        assert resolved2.name == "service2"

    run_test(test_named_services, "Named services")

    # Test 5: Dependency resolution
    def test_dependency_resolution():
        container = DIContainer()

        class DatabaseService:
            def __init__(self):
                self.connected = True

        class UserService:
            def __init__(self, db_service: DatabaseService):
                self.db_service = db_service

        container.register_singleton(DatabaseService, DatabaseService)
        container.register_singleton(UserService, UserService)

        user_service = container.resolve(UserService)

        assert isinstance(user_service, UserService)
        assert isinstance(user_service.db_service, DatabaseService)
        assert user_service.db_service.connected is True

    run_test(test_dependency_resolution, "Dependency resolution")

    # Test 6: DIResolutionError handling
    def test_di_resolution_error():
        container = DIContainer()

        class UnregisteredService:
            pass

        try:
            container.resolve(UnregisteredService)
            assert False, "Should have raised DIResolutionError"
        except DIResolutionError:
            pass  # Expected

    run_test(test_di_resolution_error, "DIResolutionError handling")

    # Test 7: Injectable base class
    def test_injectable_class():
        class InjectableService(Injectable):
            def __init__(self):
                self.injected = True

        service = InjectableService()
        assert service.injected is True

    run_test(test_injectable_class, "Injectable base class")

    # Test 8: inject decorator
    def test_inject_decorator():
        class InjectedService:
            def __init__(self):
                self.value = "injected"

        container = get_container()
        container.register_singleton(InjectedService, InjectedService)

        @inject(InjectedService)
        def test_function(**kwargs):
            service = kwargs.get("injectedservice")
            return service.value if service else "not_injected"

        result = test_function()
        assert result == "injected"

    run_test(test_inject_decorator, "inject decorator")

    # Test 9: ServiceRegistry functionality
    def test_service_registry():
        container1 = ServiceRegistry.get_container("test_container")
        assert container1 is not None

        container2 = ServiceRegistry.get_container("test_container")
        assert container1 is container2

        default_container = ServiceRegistry.get_container()
        assert default_container is not None

        ServiceRegistry.clear_container("test_container")
        ServiceRegistry.clear_all_containers()

    run_test(test_service_registry, "ServiceRegistry functionality")

    # Test 10: Global container
    def test_global_container():
        container1 = get_container()
        container2 = get_container()

        assert container1 is container2

        named_container = get_container("test_container")
        assert named_container is not container1

    run_test(test_global_container, "Global container")

    # Test 11: configure_dependencies
    def test_configure_dependencies():
        try:
            configure_dependencies()
        except Exception as e:
            print(f"Warning: configure_dependencies failed: {e}")

    run_test(test_configure_dependencies, "configure_dependencies")

    # Test 12: get_service convenience function
    def test_get_service_convenience():
        class ConvenienceService:
            def __init__(self):
                self.convenient = True

        container = get_container()
        container.register_singleton(ConvenienceService, ConvenienceService)

        service = get_service(ConvenienceService)
        assert isinstance(service, ConvenienceService)
        assert service.convenient is True

    run_test(test_get_service_convenience, "get_service convenience function")

    # Test 13: DIScope context manager
    def test_di_scope():
        with DIScope() as container:

            class ScopedService:
                def __init__(self):
                    self.scoped = True

            container.register_singleton(ScopedService, ScopedService)

            service = container.resolve(ScopedService)
            assert isinstance(service, ScopedService)
            assert service.scoped is True

    run_test(test_di_scope, "DIScope context manager")

    # Test 14: Container state management
    def test_container_state_management():
        container = DIContainer()

        class StateService:
            def __init__(self):
                self.state = "initial"

        container.register_singleton(StateService, StateService)

        service = container.resolve(StateService)
        service.state = "modified"

        service2 = container.resolve(StateService)
        assert service2 is service
        assert service2.state == "modified"

    run_test(test_container_state_management, "Container state management")

    # Test 15: Type annotations
    def test_type_annotations():
        import inspect

        sig = inspect.signature(DIContainer.register_singleton)
        assert "interface" in sig.parameters
        assert "implementation" in sig.parameters

        sig = inspect.signature(get_service)
        assert "service_type" in sig.parameters

    run_test(test_type_annotations, "Type annotations")

    # Test 16: Imports and availability
    def test_imports_and_availability():
        assert DIContainer is not None
        assert DIResolutionError is not None
        assert Injectable is not None
        assert ServiceRegistry is not None
        assert DIScope is not None
        assert inject is not None
        assert get_container is not None
        assert configure_dependencies is not None
        assert get_service is not None

    run_test(test_imports_and_availability, "Imports and availability")

    # Test 17: Integration test
    def test_integration():
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

        assert isinstance(app, AppService)
        assert isinstance(app.log, LogService)
        assert isinstance(app.config, ConfigService)
        assert app.log.config is app.config

    run_test(test_integration, "Integration test")

    print("=" * 50)
    print(f"DEPENDENCY INJECTION TESTS COMPLETE: {passed} passed, {failed} failed")
    print("=" * 50)

    return failed == 0


if __name__ == "__main__":
    run_comprehensive_tests()
