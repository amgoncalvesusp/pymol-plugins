"""Target and analysis labels for the compact plugin dialog."""

from __future__ import annotations

from typing import Any


def populate_target_selector(combo: Any, target_ids: tuple[str, ...]) -> None:
    combo.delete(0, "end")
    for target_id in target_ids:
        combo.insert("end", target_id)
