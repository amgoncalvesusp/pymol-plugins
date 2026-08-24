"""PyMOL-facing package; bundle validation remains independent of PyMOL."""

from .plugin import __init_plugin__, open_analysis

__all__ = ["__init_plugin__", "open_analysis"]
