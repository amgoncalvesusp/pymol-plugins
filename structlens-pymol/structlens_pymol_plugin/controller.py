"""Application controller for the compact StructLens-PyMOL panel."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .bundle_reader import BundleReader
from .visualization import PyMOLVisualization


class Controller:
    def __init__(self, command: Any) -> None:
        self.command = command
        self.reader: BundleReader | None = None
        self.visualization: PyMOLVisualization | None = None

    def open_bundle(self, path: str | Path) -> BundleReader:
        if self.visualization is not None:
            self.visualization.reset()
        if self.reader is not None:
            self.reader.close()
        self.reader = BundleReader(path)
        self.visualization = PyMOLVisualization(self.command, self.reader)
        self.visualization.load_structures()
        self.visualization.create_semantic_selections()
        self.visualization.apply_preset("Overview")
        return self.reader

    def reset(self) -> None:
        if self.visualization is not None:
            self.visualization.reset()
        if self.reader is not None:
            self.reader.close()
        self.visualization = None
        self.reader = None
