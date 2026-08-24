"""PyMOL Plugin Manager entrypoint shim."""

from structlens_pymol_plugin.plugin import __init_plugin__, open_analysis

__all__ = ["__init_plugin__", "open_analysis"]
