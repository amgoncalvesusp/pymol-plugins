"""Conservative analysis profiles layered on top of raw interaction detection."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import importlib.util
import os

try:
    from .docklens_core import VALID_TYPES
except (ImportError, ValueError):
    _core_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "docklens_core.py",
    )
    _core_spec = importlib.util.spec_from_file_location(
        "pymol_interactions_docklens_core_for_profiles",
        _core_path,
    )
    if _core_spec is None or _core_spec.loader is None:
        raise ImportError("Could not load the bundled DockLens interaction core")
    _docklens_core = importlib.util.module_from_spec(_core_spec)
    _core_spec.loader.exec_module(_docklens_core)
    VALID_TYPES = _docklens_core.VALID_TYPES

if TYPE_CHECKING:
    from .results import RunResult

VALID_ANALYSIS_PROFILES = ("complete", "ds_like")
_DS_LIKE_TYPES = frozenset(
    {
        "hbond",
        "carbon_hbond",
        "saltbridge",
        "attractive_charge",
        "charge_repulsion",
        "pipi",
        "pi_sigma",
        "pication",
        "pialkyl",
        "alkyl",
        "halogen",
        "metal",
        "water_bridge",
        "pi_sulfur",
        "pi_donor_hbond",
        "pi_anion",
        "pi_lone_pair",
        "chalcogen",
    }
)
_DS_LIKE_MAX_SALTBRIDGE_DISTANCE_A = 4.0


def normalize_analysis_profile(value) -> str:
    profile = str(value or "complete").strip().lower()
    if profile not in VALID_ANALYSIS_PROFILES:
        raise ValueError("Unknown analysis profile: %s" % profile)
    return profile


def detail_matches_profile(detail, profile) -> bool:
    profile = normalize_analysis_profile(profile)
    if profile == "complete":
        return True
    if detail.interaction_type not in _DS_LIKE_TYPES:
        return False
    if detail.interaction_type != "saltbridge":
        return True
    return (
        detail.distance_A is not None
        and detail.distance_A <= _DS_LIKE_MAX_SALTBRIDGE_DISTANCE_A
    )


def build_analysis_view(result: RunResult, profile="complete") -> RunResult:
    profile = normalize_analysis_profile(profile)
    if profile == "complete":
        return result
    details = tuple(
        detail for detail in result.details if detail_matches_profile(detail, profile)
    )
    by_pose = {}
    for detail in details:
        entry = by_pose.setdefault(
            detail.pose_id,
            {"total": 0, "key": 0, "counts": {kind: 0 for kind in VALID_TYPES}},
        )
        entry["total"] += 1
        entry["key"] += int(detail.is_key_residue)
        entry["counts"][detail.interaction_type] += 1
    empty = {"total": 0, "key": 0, "counts": {kind: 0 for kind in VALID_TYPES}}
    summaries = tuple(
        replace(
            summary,
            n_total_interactions=by_pose.get(summary.pose_id, empty)["total"],
            n_key_residue_interactions=by_pose.get(summary.pose_id, empty)["key"],
            counts=by_pose.get(summary.pose_id, empty)["counts"],
        )
        for summary in result.summaries
    )
    return replace(result, details=details, summaries=summaries)
