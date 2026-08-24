"""Schema validation shared by the reader and tests, without PyMOL imports."""

from __future__ import annotations

import json
import zipfile
from pathlib import PurePosixPath
from typing import Any

from .errors import BundleCompatibilityError, BundleError, UnsafeBundleError

BUNDLE_FORMAT = "structlens-pymol"
SUPPORTED_BUNDLE_SCHEMA_MAJOR = 1


def validate_archive(archive: zipfile.ZipFile) -> dict[str, Any]:
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise UnsafeBundleError("Bundle contains duplicate entries")
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise UnsafeBundleError(f"Unsafe bundle path: {name}")
        if path.suffix.lower() in {".py", ".pyc", ".exe", ".dll", ".so", ".sh", ".bat"}:
            raise UnsafeBundleError(f"Executable bundle payload is not allowed: {name}")
    try:
        manifest = json.loads(archive.read("manifest.json"))
    except (KeyError, json.JSONDecodeError) as error:
        raise BundleError("manifest.json is missing or malformed") from error
    if not isinstance(manifest, dict) or manifest.get("format") != BUNDLE_FORMAT:
        raise BundleError("Unsupported StructLens bundle format")
    schema = str(manifest.get("schema_version", ""))
    try:
        major = int(schema.split(".", 1)[0])
    except (ValueError, IndexError) as error:
        raise BundleError("schema_version must be MAJOR.MINOR") from error
    if major > SUPPORTED_BUNDLE_SCHEMA_MAJOR:
        raise BundleCompatibilityError(f"Unsupported future schema major: {major}")
    for key in ("analysis_id", "comparison_mode", "reference_id", "target_ids", "structures"):
        if key not in manifest:
            raise BundleError(f"Manifest is missing {key}")
    structures = manifest["structures"]
    target_ids = manifest["target_ids"]
    if not isinstance(structures, dict) or not isinstance(target_ids, list):
        raise BundleError("Manifest structures and target_ids have invalid types")
    if len(target_ids) != len(set(target_ids)):
        raise BundleError("Manifest target_ids must be unique")
    if manifest["reference_id"] in target_ids:
        raise BundleError("Manifest reference_id cannot also be a target")
    if manifest["reference_id"] not in structures or any(item not in structures for item in target_ids):
        raise BundleError("Manifest references an unknown structure ID")
    for structure_id, entry in structures.items():
        if not isinstance(entry, str) or entry not in names:
            raise BundleError(f"Missing structure entry for {structure_id}")
    return manifest
