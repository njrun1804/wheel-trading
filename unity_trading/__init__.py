"""Compatibility package for legacy imports.

All functionality now resides under ``src.unity_wheel``. This package
forwards attribute and submodule lookups to the new location.
"""

import importlib
import importlib.util
from typing import List

_spec = importlib.util.find_spec("src.unity_wheel")
__path__: List[str] = (
    list(_spec.submodule_search_locations) if _spec and _spec.submodule_search_locations else []
)


def __getattr__(name: str):
    return importlib.import_module(f"src.unity_wheel.{name}")
