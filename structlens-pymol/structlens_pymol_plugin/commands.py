"""Stable PyMOL command registration."""

from __future__ import annotations

from typing import Any

from .controller import Controller
from .interface import open_interface


def register_commands(command: Any) -> Controller:
    controller = Controller(command)
    if hasattr(command, "extend"):
        command.extend("structlens_open", lambda path: controller.open_bundle(path))
        command.extend("structlens_interface", lambda csv_path, source_path: open_interface(command, csv_path, source_path))
    return controller
