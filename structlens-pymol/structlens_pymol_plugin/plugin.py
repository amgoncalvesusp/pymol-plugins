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
    from pymol.plugins import addmenuitemqt

    addmenuitemqt("StructLens-PyMOL", lambda: open_analysis())
    addmenuitemqt("StructLens-PyMOL interfaces…", lambda: _choose_interface(cmd))


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

def _choose_interface(command):
    from pymol.Qt import QtWidgets
    from .interface import open_interface

    path, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Open StructLens interface", "", "Interface CSV (*.csv)")
    if not path:
        return
    source, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select original coordinates", "", "Coordinates (*.pdb *.ent *.cif *.mmcif)")
    if source:
        try:
            open_interface(command, path, source)
        except Exception as error:
            QtWidgets.QMessageBox.warning(None, "StructLens interface", str(error))
