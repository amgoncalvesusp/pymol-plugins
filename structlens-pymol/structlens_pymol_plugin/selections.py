"""Namespaced, shell-safe selection names for PyMOL."""

from __future__ import annotations

import re

_SAFE = re.compile(r"[^A-Za-z0-9_]+")


def safe_identifier(value: str, fallback: str = "analysis") -> str:
    cleaned = _SAFE.sub("_", value).strip("_") or fallback
    return cleaned[:32]


def selection_name(analysis_id: str, category: str, target_id: str | None = None) -> str:
    prefix = f"SL_{safe_identifier(analysis_id)}"
    if target_id:
        prefix += f"_T_{safe_identifier(target_id)}"
    return f"{prefix}_{safe_identifier(category)}"
