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

    def create_interaction_selections(self) -> dict[str, str]:
        """Create conserved/gained/lost selections from stored geometry only."""
        payload = self.reader.interactions()
        records = payload.get("differences", payload.get("interaction_differences", []))
        output: dict[str, str] = {}
        for change in ("conserved", "gained", "lost"):
            name = selection_name(self.analysis_id, self.active_target or "target", f"interactions_{change}")
            expressions: list[str] = []
            for item in records if isinstance(records, list) else []:
                if item.get("change") != change:
                    continue
                record = item.get("target_record") if change != "lost" else item.get("reference_record")
                if not isinstance(record, dict):
                    continue
                for field in ("residue_a", "residue_b"):
                    residue = record.get(field)
                    if isinstance(residue, dict):
                        expressions.append(_pymol_locator(residue))
            expression = " or ".join(dict.fromkeys(expressions)) or "none"
            if hasattr(self.cmd, "select"):
                self.cmd.select(name, expression)
            self.created_selections.append(name)
            output[change] = name
        return output

    def create_site_selections(self, site_id: str) -> dict[str, str]:
        payload = self.reader.sites()
        site = payload.get(site_id, payload.get("sites", {}).get(site_id, {})) if isinstance(payload, dict) else {}
        output: dict[str, str] = {}
        for label, structure_id in (("reference", self.reader.reference_id), ("target", self.active_target)):
            name = selection_name(self.analysis_id, self.active_target or "target", f"site_{site_id}_{label}")
            residues = site.get(label, site.get(f"{label}_residues", [])) if isinstance(site, dict) else []
            expressions = [_pymol_locator(item) for item in residues if isinstance(item, dict)]
            if hasattr(self.cmd, "select"):
                self.cmd.select(name, " or ".join(expressions) or "none")
            self.created_selections.append(name)
            output[label] = name
        return output

    def draw_displacement_vectors(self, *, minimum_magnitude: float = 0.5, top_n: int = 100) -> str:
        """Draw bundle-provided arrows; no transform or magnitude is recomputed."""
        payload = self.reader.vectors()
        vectors = payload.get("vectors", payload if isinstance(payload, list) else [])
        selected = [item for item in vectors if isinstance(item, dict) and float(item.get("magnitude_angstrom", 0.0)) >= minimum_magnitude]
        selected.sort(key=lambda item: (-float(item.get("magnitude_angstrom", 0.0)), str(item.get("reference_position", ""))))
        selected = selected[:top_n]
        object_name = selection_name(self.analysis_id, self.active_target or "target", "displacement_vectors")
        if hasattr(self.cmd, "load_cgo"):
            primitives: list[float] = []
            for item in selected:
                start = item.get("start_xyz")
                end = item.get("end_xyz")
                if not isinstance(start, list | tuple) or not isinstance(end, list | tuple) or len(start) != 3 or len(end) != 3:
                    continue
                primitives.extend([0.0, float(start[0]), float(start[1]), float(start[2]), float(end[0]), float(end[1]), float(end[2]), 0.08])
            self.cmd.load_cgo(primitives, object_name)
            self.created_objects.append(object_name)
        return object_name

    def evidence_card(self, reference_position: str, target_id: str | None = None) -> dict[str, Any] | None:
        """Return the stored evidence card; this method never calculates science."""
        payload = self.reader.evidence()
        cards = payload.get("cards", payload if isinstance(payload, list) else [])
        for card in cards if isinstance(cards, list) else []:
            if card.get("reference_position") == reference_position and (target_id is None or card.get("target_id") == target_id):
                return dict(card)
        return None

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
