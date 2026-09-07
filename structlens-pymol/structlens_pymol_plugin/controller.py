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
        reader = BundleReader(path)
        visualization = None
        try:
            visualization = PyMOLVisualization(self.command, reader)
            visualization.load_structures()
            visualization.create_semantic_selections()
            visualization.apply_preset("Overview")
        except Exception:
            try:
                if visualization is not None:
                    visualization.reset()
            finally:
                reader.close()
            raise
        self.reset()
        self.reader = reader
        self.visualization = visualization
        return reader

    def reset(self) -> None:
        if self.visualization is not None:
            self.visualization.reset()
        if self.reader is not None:
            self.reader.close()
        self.visualization = None
        self.reader = None
