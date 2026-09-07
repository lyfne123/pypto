# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Per-call namespace snapshots shared by JIT key construction and specialization."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _Namespaces:
    modules: dict[int, dict[str, Any]] = field(default_factory=dict)
    functions: dict[Any, dict[str, Any]] = field(default_factory=dict)


_ACTIVE_NAMESPACES: ContextVar[_Namespaces | None] = ContextVar("jit_source_namespaces", default=None)


@contextmanager
def capture_namespaces() -> Iterator[None]:
    """Keep namespace reads consistent within one synchronous compilation request.

    Nested helpers reuse the outer snapshot. Context-local storage keeps
    concurrent callers isolated; the snapshot is discarded even on failure.
    Only bindings are copied, not arbitrary mutable configuration objects.
    """
    if _ACTIVE_NAMESPACES.get() is not None:
        yield
        return
    token = _ACTIVE_NAMESPACES.set(_Namespaces())
    try:
        yield
    finally:
        _ACTIVE_NAMESPACES.reset(token)


def function_namespace(func: Any) -> dict[str, Any]:
    """Return captured module globals with closure bindings taking precedence."""
    snapshot = _ACTIVE_NAMESPACES.get()
    if snapshot is not None and func in snapshot.functions:
        return snapshot.functions[func]

    module_globals = getattr(func, "__globals__", {})
    if snapshot is None:
        namespace = dict(module_globals)
    else:
        module_id = id(module_globals)
        if module_id not in snapshot.modules:
            snapshot.modules[module_id] = dict(module_globals)
        namespace = dict(snapshot.modules[module_id])

    freevars = getattr(getattr(func, "__code__", None), "co_freevars", ())
    closure = getattr(func, "__closure__", None) or ()
    for name, cell in zip(freevars, closure, strict=True):
        try:
            namespace[name] = cell.cell_contents
        except ValueError:
            # An unbound cell does not introduce a usable captured binding.
            pass
    if snapshot is not None:
        snapshot.functions[func] = namespace
    return namespace
