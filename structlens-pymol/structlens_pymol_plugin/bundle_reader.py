"""Safe, data-only StructLens bundle reader."""

from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from .schema import validate_archive


class BundleReader:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.manifest: dict[str, Any]
        self._archive = zipfile.ZipFile(self.path)
        self._temporary_paths: list[Path] = []
        try:
            self.manifest = validate_archive(self._archive)
        except Exception:
            self._archive.close()
            raise

    def close(self) -> None:
        self._archive.close()
        for path in self._temporary_paths:
            path.unlink(missing_ok=True)
        self._temporary_paths.clear()

    def __enter__(self) -> "BundleReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def reference_id(self) -> str:
        return str(self.manifest["reference_id"])

    @property
    def target_ids(self) -> tuple[str, ...]:
        return tuple(str(item) for item in self.manifest["target_ids"])

    def read_json(self, entry: str) -> Any:
        return json.loads(self._archive.read(entry))

    def structure_bytes(self, structure_id: str) -> bytes:
        entry = self.manifest["structures"].get(structure_id)
        if entry is None:
            raise KeyError(structure_id)
        return self._archive.read(entry)

    def structure_entry(self, structure_id: str) -> str:
        return str(self.manifest["structures"][structure_id])

    def materialize_structure(self, structure_id: str) -> Path:
        """Materialize one validated structure into a private temporary file."""

        entry = self.structure_entry(structure_id)
        suffix = Path(entry).suffix.lower()
        if suffix not in {".pdb", ".ent", ".cif", ".mmcif"}:
            suffix = ".pdb"
        with tempfile.NamedTemporaryFile(prefix="structlens-", suffix=suffix, delete=False) as handle:
            handle.write(self.structure_bytes(structure_id))
            path = Path(handle.name)
        self._temporary_paths.append(path)
        return path

    def correspondence(self) -> list[dict[str, Any]]:
        return list(self.read_json("analysis/correspondence.json"))

    def mutations(self) -> list[dict[str, Any]]:
        return list(self.read_json("analysis/mutations.json"))

    def transforms(self) -> dict[str, Any]:
        return dict(self.read_json("transforms/transforms.json"))
