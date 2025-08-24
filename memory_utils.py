#!/usr/bin/env python3
"""
Lightweight memory utilities used by high-performance caching modules.
Provides a simple ObjectPool and a lazy_property descriptor.
"""

from functools import wraps
from threading import Lock
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class ObjectPool(Generic[T]):
    """A minimal, thread-safe object pool.

    - factory: callable that returns a new instance when the pool is empty
    - max_size: maximum number of idle objects kept in the pool
    """

    def __init__(self, factory: Callable[[], T], max_size: int = 10):
        self._factory = factory
        self._pool: list[T] = []
        self._max_size = max(0, int(max_size))
        self._lock = Lock()

    def acquire(self) -> T:
        with self._lock:
            if self._pool:
                return self._pool.pop()
        # Create outside of lock to avoid holding during factory work
        return self._factory()

    def release(self, obj: T) -> None:
        with self._lock:
            if len(self._pool) < self._max_size:
                self._pool.append(obj)
            # else drop reference, letting GC reclaim

    # Common alias used by some pool APIs
    get = acquire
    put = release


class LazyProperty:
    """Descriptor that computes once per-instance and caches the result.

    Usage:
        class Foo:
            @lazy_property
            def value(self):
                return compute()
    """

    def __init__(self, func: Callable[[object], T]):
        wraps(func)(self)
        self.func = func
        self.attr_name = func.__name__

    def __get__(self, obj, objtype=None) -> T:
        if obj is None:
            return self  # access via class
        if self.attr_name in obj.__dict__:
            return obj.__dict__[self.attr_name]
        value = self.func(obj)
        obj.__dict__[self.attr_name] = value
        return value

