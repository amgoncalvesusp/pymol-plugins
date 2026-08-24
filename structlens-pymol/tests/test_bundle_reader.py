import json
import zipfile

import pytest

from structlens_pymol_plugin.bundle_reader import BundleReader
from structlens_pymol_plugin.errors import BundleCompatibilityError, UnsafeBundleError


def _manifest() -> dict:
    return {
        "format": "structlens-pymol",
        "schema_version": "1.0",
        "analysis_id": "abc",
        "comparison_mode": "pairwise",
        "reference_id": "ref",
        "target_ids": ["target"],
        "structures": {"ref": "structures/reference.pdb", "target": "structures/target.pdb"},
    }


def _write(path, manifest, extra=None):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("structures/reference.pdb", "ATOM")
        archive.writestr("structures/target.pdb", "ATOM")
        if extra:
            archive.writestr(*extra)


def test_reader_validates_manifest_and_reads_structures(tmp_path):
    path = tmp_path / "ok.structlens-pymol"
    _write(path, _manifest())
    with BundleReader(path) as reader:
        assert reader.reference_id == "ref"
        assert reader.target_ids == ("target",)
        assert reader.structure_bytes("ref") == b"ATOM"


def test_reader_rejects_future_schema(tmp_path):
    path = tmp_path / "future.structlens-pymol"
    manifest = _manifest()
    manifest["schema_version"] = "2.0"
    _write(path, manifest)
    with pytest.raises(BundleCompatibilityError):
        BundleReader(path)


def test_reader_rejects_traversal(tmp_path):
    path = tmp_path / "unsafe.structlens-pymol"
    _write(path, _manifest(), ("../../evil.py", "x"))
    with pytest.raises(UnsafeBundleError):
        BundleReader(path)
