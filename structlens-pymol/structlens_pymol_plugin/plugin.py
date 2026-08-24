"""Lazy PyMOL plugin entrypoint and compact GUI handoff."""

from __future__ import annotations

from typing import Any

from .commands import register_commands

_controller: Any = None


def __init_plugin__(app: Any = None) -> None:
    global _controller
    try:
        from pymol import cmd
    except ImportError as error:
        raise RuntimeError("StructLens-PyMOL must be loaded inside PyMOL") from error
    _controller = register_commands(cmd)
    if app is not None and hasattr(app, "addmenuitemqt"):
        app.addmenuitemqt("StructLens-PyMOL", lambda: open_analysis())


def open_analysis(path: str | None = None) -> Any:
    if _controller is None:
        raise RuntimeError("Initialize StructLens-PyMOL from the PyMOL plugin menu first")
    if path is None:
        try:
            import tkinter.filedialog as filedialog

            path = filedialog.askopenfilename(filetypes=[("StructLens bundles", "*.structlens-pymol")])
        except ImportError as error:
            raise RuntimeError("PyMOL GUI support is unavailable") from error
    if not path:
        return None
    return _controller.open_bundle(path)
