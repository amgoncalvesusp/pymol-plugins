"""Visualization adapter that consumes stored bundle values only."""

from __future__ import annotations

import re
from typing import Any

from .bundle_reader import BundleReader
from .colors import COLORS
from .selections import selection_name


class PyMOLVisualization:
    def __init__(self, command: Any, reader: BundleReader) -> None:
        self.cmd = command
        self.reader = reader
        self.analysis_id = str(reader.manifest["analysis_id"])
        self.created_objects: list[str] = []
        self.created_selections: list[str] = []
        self.active_target: str | None = self.reader.target_ids[0] if self.reader.target_ids else None

    def load_structures(self) -> tuple[str, ...]:
        objects: list[str] = []
        for index, structure_id in enumerate((self.reader.reference_id, *self.reader.target_ids)):
            object_name = selection_name(self.analysis_id, "REF" if index == 0 else f"T{index:03d}")
            structure_text = self.reader.structure_bytes(structure_id).decode("utf-8")
            if hasattr(self.cmd, "read_pdbstr"):
                self.cmd.read_pdbstr(structure_text, object_name)
            elif hasattr(self.cmd, "loadstr"):
                self.cmd.loadstr(structure_text, object_name)
            elif hasattr(self.cmd, "load"):
                self.cmd.load(str(self.reader.materialize_structure(structure_id)), object_name)
            self.created_objects.append(object_name)
            objects.append(object_name)
        self.apply_transforms(objects)
        return tuple(objects)

    def apply_transforms(self, objects: list[str] | tuple[str, ...]) -> None:
        transforms = self.reader.transforms()
        for object_name, structure_id in zip(
            objects, (self.reader.reference_id, *self.reader.target_ids), strict=True
        ):
            transform = transforms.get(structure_id) or transforms.get("target")
            if not transform or not hasattr(self.cmd, "transform_object"):
                continue
            rotation = transform.get("rotation")
            translation = transform.get("translation")
            if rotation is None or translation is None:
                continue
            matrix = (
                [value for row in rotation for value in row]
                + list(translation)
                + [0.0, 0.0, 0.0, 1.0]
            )
            self.cmd.transform_object(object_name, matrix)

    def create_semantic_selections(self) -> dict[str, str]:
        selections: dict[str, str] = {}
        rows = self.reader.correspondence()
        for category, predicate in (
            ("mutations", lambda row: row.get("status") == "substitution"),
            ("key_residues", lambda row: row.get("is_key_residue", False)),
            ("outliers", lambda row: row.get("is_outlier", False)),
            ("insertions", lambda row: row.get("status") == "insertion"),
            ("deletions", lambda row: row.get("status") == "deletion"),
        ):
            name = selection_name(self.analysis_id, category)
            expressions = [
                _pymol_locator(row.get("target"))
                for row in rows
                if predicate(row) and row.get("target")
            ]
            if hasattr(self.cmd, "select"):
                self.cmd.select(name, " or ".join(expressions) or "none")
            self.created_selections.append(name)
            selections[category] = name
        return selections

    def set_active_target(self, target_id: str) -> None:
        if target_id not in self.reader.target_ids:
            raise KeyError(target_id)
        self.active_target = target_id

    def focus_selection(self, category: str = "mutations") -> str:
        name = selection_name(self.analysis_id, category, self.active_target)
        if hasattr(self.cmd, "orient"):
            self.cmd.orient(name)
        if hasattr(self.cmd, "zoom"):
            self.cmd.zoom(name)
        return name

    def export_image(
        self,
        path: str,
        *,
        width: int = 2400,
        height: int = 1800,
        dpi: int = 300,
        ray: bool = True,
    ) -> None:
        if dpi not in {300, 600}:
            raise ValueError("Publication export supports 300 or 600 dpi")
        if ray and hasattr(self.cmd, "ray"):
            self.cmd.ray(width, height)
        if hasattr(self.cmd, "png"):
            self.cmd.png(path, width, height, dpi=dpi, ray=0)

    def apply_preset(self, name: str = "Overview") -> None:
        if hasattr(self.cmd, "color"):
            self.cmd.color(COLORS["reference"], selection_name(self.analysis_id, "REF"))
        if name.lower().startswith("mutation") and hasattr(self.cmd, "show"):
            self.cmd.show("sticks", selection_name(self.analysis_id, "mutations"))

    def reset(self) -> None:
        for name in (*self.created_selections, *self.created_objects):
            if hasattr(self.cmd, "delete"):
                self.cmd.delete(name)
        self.created_selections.clear()
        self.created_objects.clear()


def _pymol_locator(residue: dict[str, Any]) -> str:
    chain = str(residue.get("chain_id", ""))
    number = str(residue.get("auth_seq_id", ""))
    insertion = str(residue.get("insertion_code") or "")
    if (
        not re.fullmatch(r"[A-Za-z0-9_.-]+", chain)
        or not re.fullmatch(r"-?\d+", number)
        or not re.fullmatch(r"[A-Za-z]?", insertion)
    ):
        return "none"
    return f"chain {chain} and resi {number}{insertion}"
