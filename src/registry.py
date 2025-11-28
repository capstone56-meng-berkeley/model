"""Base registry pattern for extensible components."""

from typing import Dict, List, Type, TypeVar, Optional

T = TypeVar('T')


class Registry:
    """
    Generic registry for extensible components.

    Usage:
        class MyRegistry(Registry):
            _registry = {}

        @MyRegistry.register("my_component")
        class MyComponent:
            ...

        # Get component
        cls = MyRegistry.get("my_component")

        # List available
        MyRegistry.list_available()  # ["my_component"]
    """
    _registry: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a new component.

        Args:
            name: Unique identifier for the component

        Returns:
            Decorator function
        """
        def decorator(component_cls: Type[T]) -> Type[T]:
            if name in cls._registry:
                raise ValueError(
                    f"Component '{name}' already registered in {cls.__name__}. "
                    f"Existing: {cls._registry[name].__name__}"
                )
            cls._registry[name] = component_cls
            return component_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[Type]:
        """
        Get a component by name.

        Args:
            name: Component identifier

        Returns:
            Component class or None if not found
        """
        return cls._registry.get(name)

    @classmethod
    def get_or_raise(cls, name: str) -> Type:
        """
        Get a component by name, raising if not found.

        Args:
            name: Component identifier

        Returns:
            Component class

        Raises:
            ValueError: If component not found
        """
        component = cls._registry.get(name)
        if component is None:
            available = cls.list_available()
            raise ValueError(
                f"Unknown component '{name}' in {cls.__name__}. "
                f"Available: {available}"
            )
        return component

    @classmethod
    def list_available(cls) -> List[str]:
        """
        List all registered component names.

        Returns:
            List of component identifiers
        """
        return list(cls._registry.keys())

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> T:
        """
        Create an instance of a registered component.

        Args:
            name: Component identifier
            *args: Positional arguments for constructor
            **kwargs: Keyword arguments for constructor

        Returns:
            Component instance
        """
        component_cls = cls.get_or_raise(name)
        return component_cls(*args, **kwargs)

    @classmethod
    def clear(cls) -> None:
        """Clear all registered components (mainly for testing)."""
        cls._registry = {}
