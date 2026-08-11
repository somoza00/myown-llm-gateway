"""Smoke tests: every module under src/llm_gateway must import cleanly."""

import importlib
import pkgutil

import llm_gateway

MODULES = [m.name for m in pkgutil.walk_packages(llm_gateway.__path__, "llm_gateway.")]


def test_package_has_modules() -> None:
    assert MODULES


def test_all_modules_import() -> None:
    for name in MODULES:
        importlib.import_module(name)
