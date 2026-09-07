"""PyMOL-facing package; bundle validation remains independent of PyMOL."""

__version__ = "0.3.1"

from .plugin import __init_plugin__, open_analysis

__all__ = ["__init_plugin__", "open_analysis", "__version__"]
