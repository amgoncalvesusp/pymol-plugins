"""Stable PyMOL command registration."""

from __future__ import annotations

from typing import Any

from .controller import Controller


def register_commands(command: Any) -> Controller:
    controller = Controller(command)
    if hasattr(command, "extend"):
        command.extend("structlens_open", lambda path: controller.open_bundle(path))
    return controller
