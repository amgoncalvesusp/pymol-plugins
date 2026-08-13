"""
interaction_core.py — PyMOL-independent non-covalent interaction detection.

The legacy PLIP branch is ported from the PyMOL plugin
``interactions_plugin.py`` so historical results remain reproducible. Native
LUNA, DSV-like and conservative LUNA × DSV profiles add chemistry-aware
geometry without changing the legacy contract.

  * ``Atom`` is built from plain fields (not a PyMOL chempy atom) and gains
    parser/UI bookkeeping fields (serial, subst_id, side).
  * ``Atom.res_sele()`` (the only PyMOL-coupled method) is removed; ``res_tag``
    and ``label`` are kept.
  * Interaction dictionaries carry endpoint objects (``a_obj``/``b_obj``) instead
    of PyMOL selection strings (``a_sele``/``b_sele``), so the desktop tool can
    resolve ligand vs. receptor side per endpoint.
"""

from __future__ import annotations

from contextvars import ContextVar
from types import MappingProxyType

import numpy as np

# ===========================================================================
# Colour palette (Okabe-Ito) — shared with the PyMOL plugin for visual parity
# ===========================================================================

# Okabe & Ito (2008) colour-blind-safe qualitative palette. RGB 0-255.
_OKABE_ITO = {
    "black": (0, 0, 0),
    "orange": (230, 159, 0),
    "skyblue": (86, 180, 233),
    "bluishgreen": (0, 158, 115),
    "yellow": (240, 228, 66),
    "blue": (0, 114, 178),
    "vermillion": (213, 94, 0),
    "reddishpurple": (204, 121, 167),
}

# type -> (okabe-ito colour key, human label)
INTERACTION_COLORS = {
    "hbond": ("skyblue", "Hydrogen bond (conventional)"),
    "carbon_hbond": ("bluishgreen", "Carbon H-bond (weak, C-H...O/N)"),
    "saltbridge": ("vermillion", "Salt bridge / ionic"),
    "attractive_charge": ("orange", "Attractive charge interaction"),
    "charge_repulsion": ("vermillion", "Repulsive charge interaction"),
    "pipi": ("reddishpurple", "pi-pi stacking (sandwich/T-shaped)"),
    "pication": ("yellow", "pi-cation"),
    "pialkyl": ("orange", "pi-alkyl"),
    "pi_sigma": ("reddishpurple", "pi-sigma"),
    "alkyl": ("blue", "Alkyl-alkyl (hydrophobic)"),
    "halogen": ("black", "Halogen bond"),
    "metal": ("vermillion", "Metal coordination"),
    "water_bridge": ("skyblue", "Water-mediated H-bond"),
    "pi_sulfur": ("reddishpurple", "pi-sulfur"),
    "pi_anion": ("yellow", "pi-anion"),
    "pi_donor_hbond": ("bluishgreen", "pi-donor hydrogen bond"),
    "pi_lone_pair": ("skyblue", "Lone pair-pi"),
    "chalcogen": ("black", "Chalcogen bond (S/Se/Te)"),
}


def color_hex(itype: str) -> str:
    """Return the '#RRGGBB' Okabe-Ito colour for an interaction type."""
    okabe_key = INTERACTION_COLORS[itype][0]
    r, g, b = _OKABE_ITO[okabe_key]
    return "#%02X%02X%02X" % (r, g, b)


# ===========================================================================
# Geometric cutoffs: legacy defaults plus separately gated strict criteria.
# ===========================================================================
#
# Sources:
#   PLIP = Salentin et al. 2015 and the historical companion plug-in.
#   LUNA = keiserlab/LUNA 0.14 ``luna/interaction/config.cfg``.
#   DSV  = values queried from Discovery Studio Visualizer 2024
#          ``Mdm::NonbondMonitor`` (24.1.0.23298).
#
CUTOFFS = {
    "hbond_dist": 4.1,
    "hbond_angle": 100.0,
    "carbon_hbond_dist": 3.6,  # UNCERTAIN
    "carbon_hbond_angle": 120.0,  # UNCERTAIN
    "saltbridge_dist": 5.5,
    "pipi_dist": 5.5,
    "pipi_offset": 2.0,
    "pipi_angle_dev": 30.0,
    "pication_dist": 6.0,
    "pication_offset": 2.0,
    "pialkyl_dist": 5.0,  # legacy profile
    "alkyl_dist": 4.0,
    "halogen_dist": 4.0,
    "halogen_angle": 135.0,
    "metal_dist": 3.0,
    "water_bridge_min": 2.5,
    "water_bridge_max": 4.1,
    "water_bridge_angle_min": 75.0,
    "water_bridge_angle_max": 140.0,
    "pi_sigma_carbon_dist": 4.5,
    "pi_sigma_h_centroid_dist": 4.3,
    "pi_sigma_axis_angle": 40.0,
    "pi_sigma_dha_angle": 160.0,
    "pi_donor_dist": 5.2,
    "pi_donor_h_centroid_dist": 4.1,
    "pi_donor_axis_angle": 45.0,
    "pi_donor_dha_angle": 145.0,
    "pi_sulfur_dist": 5.3,  # UNCERTAIN
    "pi_anion_dist": 5.0,  # UNCERTAIN
    "pi_anion_offset": 2.0,
}

_CUTOFF_DEFAULTS = dict(CUTOFFS)

# This public name is retained for project/manifest compatibility even though
# each entry now configures the complete interaction detector, not only H-bonds.
_PROFILE_OVERRIDES = {
    "plip": {
        "hbond_dist": _CUTOFF_DEFAULTS["hbond_dist"],
        "hbond_angle": _CUTOFF_DEFAULTS["hbond_angle"],
        "carbon_hbond_dist": _CUTOFF_DEFAULTS["carbon_hbond_dist"],
        "carbon_hbond_angle": _CUTOFF_DEFAULTS["carbon_hbond_angle"],
    },
    "luna": {
        "hbond_dist": 3.9,
        "hbond_angle": 90.0,
        "hbond_h_a_dist": 2.8,
        "hbond_acceptor_angle": 90.0,
        "hbond_dar_angle": 90.0,
        "hbond_h_a_required": 1.0,
        "carbon_hbond_dist": 4.0,
        "carbon_hbond_angle": 110.0,
        "carbon_hbond_h_a_dist": 3.0,
        "carbon_hbond_acceptor_angle": 90.0,
        "carbon_hbond_dar_angle": 90.0,
        "carbon_hbond_h_a_required": 1.0,
        "saltbridge_dist": 6.0,
        "attractive_charge_dist": 6.0,
        "repulsive_charge_dist": 6.0,
        "pipi_dist": 6.0,
        "pipi_stacked_theta_max": 30.0,
        "pipi_t_theta_deviation_max": 30.0,
        "pipi_slope_angle_min": 30.0,
        "pipi_slope_angle_max": 60.0,
        "pipi_offset_angle_min": 30.0,
        "pipi_offset_angle_max": 60.0,
        "pication_dist": 6.0,
        "pialkyl_dist": 4.5,
        "alkyl_dist": 4.5,
        "halogen_dist": 4.0,
        "halogen_ring_dist": 4.5,
        "halogen_donor_angle_min": 120.0,
        "halogen_acceptor_angle_min": 80.0,
        "halogen_displacement_angle_max": 60.0,
        "chalcogen_dist": 4.0,
        "chalcogen_ring_dist": 4.5,
        "chalcogen_donor_angle_min": 120.0,
        "chalcogen_acceptor_angle_min": 80.0,
        "chalcogen_displacement_angle_max": 60.0,
        "pi_sulfur_dist": 4.5,
        "metal_dist": 2.8,
    },
    "dsv": {
        "hbond_dist": 3.4,
        "hbond_angle": 90.0,
        "hbond_acceptor_angle": 90.0,
        "hbond_dar_angle": 90.0,
        "hbond_h_a_required": 0.0,
        "carbon_hbond_dist": 3.8,
        "carbon_hbond_angle": 90.0,
        "carbon_hbond_acceptor_angle": 90.0,
        "carbon_hbond_dar_angle": 90.0,
        "carbon_hbond_h_a_required": 0.0,
        "saltbridge_dist": 4.0,
        "charge_dist": 5.6,
        "attractive_charge_dist": 5.6,
        "repulsive_charge_dist": 5.6,
        "pipi_dist": 6.0,
        "pipi_closest_atom_dist": 4.5,
        "pipi_stacked_theta_max": 50.0,
        "pipi_stacked_gamma_max": 35.0,
        "pipi_t_theta_deviation_max": 30.0,
        "pipi_t_gamma_min": 55.0,
        "pication_dist": 5.0,
        "pication_angle_max": 40.0,
        "pialkyl_dist": 5.5,
        "alkyl_dist": 5.5,
        "halogen_f_dist": 3.7,
        "halogen_vdw_fraction": 1.0,
        "halogen_donor_angle_min": 120.0,
        "halogen_acceptor_angle_min": 75.0,
        "metal_dist": 3.0,
        "pi_sigma_carbon_dist": 4.0,
        "pi_sigma_axis_angle": 45.0,
        "pi_sigma_dha_angle": 160.0,
        "pi_sigma_use_h_centroid": 0.0,
        "pi_donor_dist": 4.2,
        "pi_donor_axis_angle": 45.0,
        "pi_donor_dha_angle": 140.0,
        "pi_donor_use_h_centroid": 0.0,
        "pi_sulfur_edge_dist": 6.0,
        "pi_sulfur_edge_angle_min": 70.0,
        "pi_sulfur_face_dist": 4.5,
        "pi_sulfur_face_angle_max": 25.0,
        "pi_sulfur_dist": 6.0,
        "pi_lone_pair_dist": 3.0,
        "pi_lone_pair_angle": 45.0,
        "chalcogen_vdw_fraction": 1.0,
        "chalcogen_donor_angle_min": 90.0,
        "chalcogen_acceptor_angle_min": 60.0,
    },
}


def _conservative_luna_dsv_overrides():
    """Return the strict overlap plus the native-only criteria of each source."""
    dsv = _PROFILE_OVERRIDES["dsv"]
    luna = _PROFILE_OVERRIDES["luna"]
    values = dict(dsv)
    values.update(
        {
            "hbond_dist": min(dsv["hbond_dist"], luna["hbond_dist"]),
            "hbond_h_a_dist": luna["hbond_h_a_dist"],
            "hbond_h_a_required": 1.0,
            "hbond_angle": max(dsv["hbond_angle"], luna["hbond_angle"]),
            "hbond_acceptor_angle": max(
                dsv["hbond_acceptor_angle"], luna["hbond_acceptor_angle"]
            ),
            "carbon_hbond_dist": min(
                dsv["carbon_hbond_dist"], luna["carbon_hbond_dist"]
            ),
            "carbon_hbond_h_a_dist": luna["carbon_hbond_h_a_dist"],
            "carbon_hbond_h_a_required": 1.0,
            "carbon_hbond_angle": max(
                dsv["carbon_hbond_angle"], luna["carbon_hbond_angle"]
            ),
            "carbon_hbond_acceptor_angle": max(
                dsv["carbon_hbond_acceptor_angle"],
                luna["carbon_hbond_acceptor_angle"],
            ),
            "saltbridge_dist": min(
                dsv["saltbridge_dist"], luna["saltbridge_dist"]
            ),
            "attractive_charge_dist": min(
                dsv["attractive_charge_dist"],
                luna["attractive_charge_dist"],
            ),
            "repulsive_charge_dist": min(
                dsv["repulsive_charge_dist"], luna["repulsive_charge_dist"]
            ),
            "pipi_dist": min(dsv["pipi_dist"], luna["pipi_dist"]),
            "pipi_stacked_theta_max": min(
                dsv["pipi_stacked_theta_max"],
                luna["pipi_stacked_theta_max"],
            ),
            "pipi_stacked_gamma_max": 30.0,
            "pipi_t_theta_deviation_max": min(
                dsv["pipi_t_theta_deviation_max"],
                luna["pipi_t_theta_deviation_max"],
            ),
            "pipi_t_gamma_min": 60.0,
            "pication_dist": min(dsv["pication_dist"], luna["pication_dist"]),
            "alkyl_dist": min(dsv["alkyl_dist"], luna["alkyl_dist"]),
            "halogen_dist": luna["halogen_dist"],
            "halogen_acceptor_angle_min": max(
                dsv["halogen_acceptor_angle_min"],
                luna["halogen_acceptor_angle_min"],
            ),
            "halogen_donor_angle_min": max(
                dsv["halogen_donor_angle_min"],
                luna["halogen_donor_angle_min"],
            ),
            "chalcogen_dist": luna["chalcogen_dist"],
            "chalcogen_donor_angle_min": luna["chalcogen_donor_angle_min"],
            "chalcogen_acceptor_angle_min": luna[
                "chalcogen_acceptor_angle_min"
            ],
            "metal_dist": min(dsv["metal_dist"], luna["metal_dist"]),
            "pi_sulfur_dist": min(
                dsv["pi_sulfur_face_dist"], luna["pi_sulfur_dist"]
            ),
        }
    )
    return values


_PROFILE_OVERRIDES["luna_dsv"] = _conservative_luna_dsv_overrides()
HBOND_PRESETS = MappingProxyType(
    {
        name: MappingProxyType(dict(_PROFILE_OVERRIDES[name]))
        for name in ("plip", "luna", "dsv", "luna_dsv")
    }
)

_NEW_PROFILE_TYPES = {"attractive_charge", "charge_repulsion", "chalcogen"}
_LEGACY_TYPES = tuple(
    kind for kind in INTERACTION_COLORS if kind not in _NEW_PROFILE_TYPES
)
_PROFILE_DEFAULT_TYPES = MappingProxyType(
    {
        "plip": _LEGACY_TYPES,
        "luna": (
            "hbond",
            "carbon_hbond",
            "attractive_charge",
            "charge_repulsion",
            "pipi",
            "pication",
            "alkyl",
            "halogen",
            "metal",
            "pi_sulfur",
            "chalcogen",
        ),
        "dsv": tuple(INTERACTION_COLORS),
        "luna_dsv": tuple(INTERACTION_COLORS),
    }
)


def default_types_for_profile(name):
    """Return the non-inflating default interaction families for a profile."""
    normalized = str(name).strip().lower()
    if normalized not in _PROFILE_DEFAULT_TYPES:
        raise ValueError("Unknown chemistry profile: %s" % normalized)
    return _PROFILE_DEFAULT_TYPES[normalized]


def apply_hbond_preset(name):
    """Reset globals and apply one named scientific profile in-place."""
    preset = HBOND_PRESETS.get(str(name).lower())
    if preset:
        CUTOFFS.clear()
        CUTOFFS.update(_CUTOFF_DEFAULTS)
        CUTOFFS.update(preset)


def cutoffs_for_preset(name):
    """Return an immutable profile snapshot without mutating process globals."""
    normalized = str(name).strip().lower()
    if normalized not in HBOND_PRESETS:
        raise ValueError("Unknown chemistry profile: %s" % normalized)
    values = dict(_CUTOFF_DEFAULTS)
    values.update(HBOND_PRESETS[normalized])
    return MappingProxyType(values)


_ACTIVE_CUTOFFS = ContextVar(
    "docklens_active_cutoffs", default=MappingProxyType(dict(CUTOFFS))
)
_ACTIVE_CHEMISTRY_PROFILE = ContextVar(
    "docklens_active_chemistry_profile", default="plip"
)


def _active_cutoffs():
    return _ACTIVE_CUTOFFS.get()


VALID_TYPES = list(INTERACTION_COLORS.keys())

# Planarity tolerance for aromatic-ring perception (Angstrom). Verbatim.
_RING_PLANARITY_TOL = 0.15
_RING_ELEMENTS = {"C", "N", "O", "S"}


# ===========================================================================
# Small vector helpers — VERBATIM
# ===========================================================================


def _v(coord):
    return np.asarray(coord, dtype=float)


def _dist(a, b):
    return float(np.linalg.norm(_v(a) - _v(b)))


def _centroid(coords):
    return np.mean(np.asarray(coords, dtype=float), axis=0)


def _plane_normal(coords):
    """Best-fit plane normal via SVD of centred coordinates."""
    pts = np.asarray(coords, dtype=float)
    centred = pts - pts.mean(axis=0)
    _u, _s, vh = np.linalg.svd(centred)
    return vh[2]


def _planar_deviation(coords):
    """Max absolute distance of any atom from the best-fit plane."""
    pts = np.asarray(coords, dtype=float)
    n = _plane_normal(pts)
    return float(np.max(np.abs((pts - pts.mean(axis=0)).dot(n))))


def _angle_at(vertex, p1, p2):
    """Angle p1-vertex-p2 in degrees."""
    a = _v(p1) - _v(vertex)
    b = _v(p2) - _v(vertex)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    cosv = np.clip(a.dot(b) / (na * nb), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosv)))


def _plane_angle(n1, n2):
    """Acute angle (deg) between two plane normals."""
    cosv = abs(np.clip(_v(n1).dot(_v(n2)), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosv)))


def _proj_offset(point, plane_point, normal):
    """Lateral offset of `point` from the axis through `plane_point`."""
    d = _v(point) - _v(plane_point)
    along = d.dot(_v(normal))
    perp = d - along * _v(normal)
    return float(np.linalg.norm(perp))


# ===========================================================================
# Atom / topology — ported (Atom rebuilt from plain fields; res_sele removed)
# ===========================================================================


class Atom(object):
    """Lightweight atom. Geometry-relevant fields match the plugin's Atom;
    parser-provided chemistry fields are optional so non-MOL2 formats and
    callers from older releases retain their previous behaviour."""

    __slots__ = (
        "idx",
        "elem",
        "name",
        "resn",
        "resi",
        "chain",
        "segi",
        "coord",
        "fcharge",
        "neighbors",
        "serial",
        "subst_id",
        "side",
        "sybyl_type",
        "partial_charge",
        "bond_orders",
    )

    def __init__(
        self,
        idx,
        elem,
        name,
        resn,
        resi,
        chain="",
        segi="",
        coord=(0.0, 0.0, 0.0),
        fcharge=0,
        serial=None,
        subst_id=None,
        sybyl_type="",
        partial_charge=None,
    ):
        self.idx = idx
        self.elem = (elem or (name[:1] if name else "")).strip().capitalize()
        self.name = (name or "").strip()
        self.resn = (resn or "").strip()
        self.resi = str(resi).strip()
        self.chain = (chain or "").strip()
        self.segi = (segi or "").strip()
        self.coord = _v(coord)
        try:
            self.fcharge = int(fcharge)
        except Exception:
            self.fcharge = 0
        self.neighbors = []
        self.serial = serial
        self.subst_id = subst_id
        self.side = None  # 'receptor' | 'ligand' | 'water', set before classify
        self.sybyl_type = (sybyl_type or "").strip()
        try:
            self.partial_charge = (
                float(partial_charge) if partial_charge is not None else None
            )
        except (TypeError, ValueError):
            self.partial_charge = None
        # neighbor atom index -> raw, normalized MOL2 bond type
        self.bond_orders = {}

    # --- kept from the plugin ---
    def res_tag(self):
        chain = self.chain if self.chain else "_"
        return "%s%s%s" % (self.resn, self.resi, ("" if chain == "_" else chain))

    def label(self):
        return "%s_%s" % (self.res_tag(), self.name)


def _find_rings(atoms):
    """Topology-based aromatic-ring perception. VERBATIM."""
    adj = {a.idx: [n.idx for n in a.neighbors] for a in atoms}
    by_idx = {a.idx: a for a in atoms}
    rings = set()
    for start in adj:
        stack = [(start, (start,))]
        while stack:
            cur, path = stack.pop()
            for nb in adj[cur]:
                if nb == start and len(path) >= 5:
                    rings.add(frozenset(path))
                elif nb not in path and len(path) < 6:
                    stack.append((nb, path + (nb,)))

    ring_atoms = []
    for ring in rings:
        members = [by_idx[i] for i in ring]
        if not (5 <= len(members) <= 6):
            continue
        if any(a.elem not in _RING_ELEMENTS for a in members):
            continue
        if _planar_deviation([a.coord for a in members]) > _RING_PLANARITY_TOL:
            continue
        ring_atoms.append(members)
    return ring_atoms


class Ring(object):
    """Aromatic ring feature: centroid, plane normal, label. VERBATIM."""

    __slots__ = ("atoms", "centroid", "normal", "tag")

    def __init__(self, members, tag):
        self.atoms = members
        coords = [a.coord for a in members]
        self.centroid = _centroid(coords)
        self.normal = _plane_normal(coords)
        self.tag = tag


def _build_rings(atoms):
    """VERBATIM."""
    rings = _find_rings(atoms)
    per_res = {}
    for members in rings:
        per_res[members[0].res_tag()] = per_res.get(members[0].res_tag(), 0) + 1
    counters = {}
    out = []
    for members in rings:
        res = members[0].res_tag()
        if per_res[res] > 1:
            counters[res] = counters.get(res, 0) + 1
            tag = "%s_ring%d" % (res, counters[res])
        else:
            tag = "%s_ring" % res
        out.append(Ring(members, tag))
    return out


# ===========================================================================
# Chemical feature classification — ported (charged tuples carry the atom obj)
# ===========================================================================

_CATION_RES_ATOMS = {
    "LYS": ["NZ"],
    "ARG": ["NH1", "NH2", "NE"],
    "HIS": ["ND1", "NE2"],
    "HIP": ["ND1", "NE2"],
    "HSP": ["ND1", "NE2"],
}
_ANION_RES_ATOMS = {
    "ASP": ["OD1", "OD2"],
    "GLU": ["OE1", "OE2"],
}
_HALOGENS = {"F", "Cl", "Br", "I"}
_CHALCOGENS = {"S", "Se", "Te"}
_HB_ACCEPTOR_ELEMS = {"N", "O", "S", "F"}
_HB_DONOR_ELEMS = {"N", "O"}
_METALS = {"Na", "K", "Mg", "Ca", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Cd", "Hg"}
_WATER_RESN = {"HOH", "WAT", "H2O", "SOL", "TIP", "TIP3", "TIP4", "SPC", "DOD"}

# Bondi/commonly used van der Waals radii (angstrom). Only elements required by
# the DSV halogen/sulfur criteria are included; unknowns use a conservative
# 1.70 A fallback rather than silently bypassing the VDW-fraction criterion.
_VDW_RADII = {
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "Cl": 1.75,
    "Se": 1.90,
    "Br": 1.85,
    "Te": 2.06,
    "I": 1.98,
}


def _h_neighbors(atom):
    return [n for n in atom.neighbors if n.elem == "H"]


_CHEM_NON_ACCEPTOR_SYBYL = {"n.am", "n.4", "n.pl3"}
_CHEM_NON_DONOR_SYBYL = {"o.2", "o.co2"}
_PROTEIN_NON_ACCEPTOR = {
    ("ARG", "NE"),
    ("ARG", "NH1"),
    ("ARG", "NH2"),
    ("ASN", "ND2"),
    ("GLN", "NE2"),
    ("LYS", "NZ"),
}
_PROTEIN_NON_DONOR_OXYGEN = {
    ("ASP", "OD1"),
    ("ASP", "OD2"),
    ("GLU", "OE1"),
    ("GLU", "OE2"),
}
_PROTEIN_HYDROXYL_DONORS = {
    ("SER", "OG"),
    ("THR", "OG1"),
    ("TYR", "OH"),
}


def _sybyl(atom):
    return atom.sybyl_type.lower()


def _heavy_neighbors(atom):
    return [neighbor for neighbor in atom.neighbors if neighbor.elem != "H"]


def _ring_has_aromatic_evidence(ring):
    """Require MOL2 aromatic typing/bonds for strict pi interactions."""
    members = tuple(ring.atoms)
    member_ids = {atom.idx for atom in members}
    typed_aromatic = all(_sybyl(atom) in {"c.ar", "n.ar"} for atom in members)
    bonded_aromatic = all(
        sum(
            str(atom.bond_orders.get(neighbor.idx, "")).lower() == "ar"
            for neighbor in atom.neighbors
            if neighbor.idx in member_ids
        )
        >= 2
        for atom in members
    )
    return typed_aromatic or bonded_aromatic


_PROTEIN_RESIDUES = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "ASH",
    "GLH",
    "HID",
    "HIE",
    "HIP",
    "HSP",
    "CYM",
    "CYX",
}
_PROTEIN_ALKYL_ATOMS = {
    "ALA": {"CB"},
    "VAL": {"CB", "CG1", "CG2"},
    "LEU": {"CB", "CG", "CD1", "CD2"},
    "ILE": {"CB", "CG1", "CG2", "CD1"},
    "MET": {"CB", "CG", "CE"},
    "PRO": {"CB", "CG", "CD"},
    "CYS": {"CB"},
    "CYM": {"CB"},
    "CYX": {"CB"},
}


def _chemistry_aware_alkyl_carbon(atom):
    """Recognize aliphatic groups without counting arbitrary protein C?/C?."""
    residue = atom.resn.upper()
    if residue in _PROTEIN_RESIDUES:
        return atom.name.upper() in _PROTEIN_ALKYL_ATOMS.get(residue, set())
    heavy = _heavy_neighbors(atom)
    return bool(heavy) and all(neighbor.elem == "C" for neighbor in heavy)


def _strict_group_is_cationic(resname, atoms):
    """Avoid assigning a positive centre to neutral histidine."""
    if resname != "HIS":
        return True
    if any(atom.fcharge > 0 for atom in atoms):
        return True
    return len(atoms) >= 2 and all(_h_neighbors(atom) for atom in atoms)


_BOND_ORDER_VALUES = {
    "1": 1.0,
    "2": 2.0,
    "3": 3.0,
    "ar": 1.5,
    "am": 1.0,
}


def _heavy_bond_order_sum(atom):
    total = 0.0
    for neighbor in _heavy_neighbors(atom):
        raw_order = str(atom.bond_orders.get(neighbor.idx, "1")).lower()
        total += _BOND_ORDER_VALUES.get(raw_order, 1.0)
    return total


def _chemistry_aware_acceptor(atom):
    """Conservative acceptor policy based on retained topology/atom types."""
    if atom.elem not in _HB_ACCEPTOR_ELEMS or atom.fcharge > 0:
        return False
    sybyl = _sybyl(atom)
    if sybyl in _CHEM_NON_ACCEPTOR_SYBYL:
        return False
    if sybyl == "o.3" and _h_neighbors(atom):
        return False
    if atom.elem == "N":
        residue_atom = (atom.resn.upper(), atom.name.upper())
        if atom.name.upper() == "N" or residue_atom in _PROTEIN_NON_ACCEPTOR:
            return False
    return True


def _chemistry_aware_donor(atom, allow_inferred_hydrogen=True):
    """Whether an N/O atom can carry a donor hydrogen.

    Explicit hydrogens are authoritative except on carbonyl/carboxylate
    oxygens. Without explicit H, only chemically unsaturated environments are
    admitted. This decision is deliberately per atom: an unrelated hydrogen
    elsewhere in the complex does not change it.
    """
    if atom.elem not in _HB_DONOR_ELEMS:
        return False
    sybyl = _sybyl(atom)
    if sybyl in _CHEM_NON_DONOR_SYBYL:
        return False
    if _h_neighbors(atom):
        return True
    if not allow_inferred_hydrogen:
        return False

    heavy = _heavy_neighbors(atom)
    residue_atom = (atom.resn.upper(), atom.name.upper())
    if atom.elem == "O":
        if residue_atom in _PROTEIN_NON_DONOR_OXYGEN:
            return False
        # O.3 with zero/one heavy neighbor can be an alcohol/phenol donor.
        # The same valence rule is a conservative fallback for PDB/PDBQT.
        available_valence = len(heavy) <= 1 and _heavy_bond_order_sum(atom) <= 1.0
        return (sybyl == "o.3" and available_valence) or (
            not sybyl and available_valence and residue_atom in _PROTEIN_HYDROXYL_DONORS
        )

    # Aromatic/pyridine-like and nitrile nitrogens need an explicit hydrogen
    # to be treated as donors. Quaternary nitrogens cannot accept an inferred H.
    if sybyl in {"n.1", "n.2", "n.ar", "n.4"}:
        return False
    if atom.resn.upper() == "PRO" and atom.name.upper() == "N":
        return False
    return len(heavy) < 3 and _heavy_bond_order_sum(atom) < 3.0


def classify(atoms, rings, has_h, chemistry_profile="plip"):
    """Return a dict of feature lists for one molecular side. Ported.

    The only change from the plugin: charged-centre tuples carry the
    representative Atom object as the third element (was a PyMOL selection
    string), so the caller can recover residue/side metadata.
    """
    ring_atom_ids = set(a.idx for r in rings for a in r.atoms)

    donors = []
    carbon_donors = []
    acceptors = []
    sigma_donors = []
    cations = []  # (point, label, repr_atom)
    anions = []  # (point, label, repr_atom)
    halogens = []
    alkyl_carbons = []
    metals = []
    sulfurs = []
    chalcogens = []
    profile = str(chemistry_profile).strip().lower()
    chemistry_aware = profile != "plip"
    requires_explicit_donor = profile in {"luna", "luna_dsv"}
    side_has_explicit_hydrogens = any(atom.elem == "H" for atom in atoms)

    # charged centres from formal charge
    for a in atoms:
        if a.fcharge > 0:
            cations.append((a.coord, a.label(), a))
        elif a.fcharge < 0:
            anions.append((a.coord, a.label(), a))

    # protein charged groups (grouped centres)
    grouped_cation = {}
    grouped_anion = {}
    for a in atoms:
        if a.resn in _CATION_RES_ATOMS and a.name in _CATION_RES_ATOMS[a.resn]:
            grouped_cation.setdefault((a.res_tag(), a.resn), []).append(a)
        if a.resn in _ANION_RES_ATOMS and a.name in _ANION_RES_ATOMS[a.resn]:
            grouped_anion.setdefault((a.res_tag(), a.resn), []).append(a)
    for (res, resn), grp in grouped_cation.items():
        if chemistry_aware and not _strict_group_is_cationic(resn, grp):
            continue
        pt = _centroid([x.coord for x in grp])
        lbl = "%s_guan" % res if resn == "ARG" else "%s_%s" % (res, grp[0].name)
        cations.append((pt, lbl, grp[0]))
    for (res, _resn), grp in grouped_anion.items():
        pt = _centroid([x.coord for x in grp])
        anions.append((pt, "%s_carboxyl" % res, grp[0]))

    # MOL2 partial charges are not formal charges, but selected SYBYL types
    # encode formal ionic states unambiguously enough for grouped centres.
    if chemistry_aware:
        grouped_cation_atoms = {
            atom.idx for group in grouped_cation.values() for atom in group
        }
        grouped_anion_atoms = {
            atom.idx for group in grouped_anion.values() for atom in group
        }
        for atom in atoms:
            if (
                _sybyl(atom) == "n.4"
                and atom.fcharge <= 0
                and atom.idx not in grouped_cation_atoms
            ):
                cations.append((atom.coord, atom.label(), atom))

        sybyl_carboxylates = {}
        for atom in atoms:
            if (
                _sybyl(atom) != "o.co2"
                or atom.fcharge < 0
                or atom.idx in grouped_anion_atoms
            ):
                continue
            carbon_neighbors = [
                neighbor for neighbor in _heavy_neighbors(atom) if neighbor.elem == "C"
            ]
            group_key = (
                ("carbon", carbon_neighbors[0].idx)
                if carbon_neighbors
                else ("oxygen", atom.idx)
            )
            sybyl_carboxylates.setdefault(group_key, []).append(atom)
        for group in sybyl_carboxylates.values():
            point = _centroid([atom.coord for atom in group])
            anions.append((point, "%s_carboxylate" % group[0].res_tag(), group[0]))

    # H-bond donors/acceptors, halogens, alkyl carbons, metals, sulfurs
    for a in atoms:
        if chemistry_aware:
            if _chemistry_aware_acceptor(a):
                acceptors.append(a)
            if _chemistry_aware_donor(
                a,
                allow_inferred_hydrogen=(
                    not requires_explicit_donor
                    and not side_has_explicit_hydrogens
                ),
            ):
                hs = _h_neighbors(a)
                donors.append((a, hs))
        else:
            if a.elem in _HB_ACCEPTOR_ELEMS:
                if not (a.elem == "N" and a.fcharge > 0):
                    acceptors.append(a)
            if a.elem in _HB_DONOR_ELEMS:
                if has_h:
                    hs = _h_neighbors(a)
                    if hs:
                        donors.append((a, hs))
                else:
                    donors.append((a, []))
        if a.elem == "C":
            if has_h:
                hs = _h_neighbors(a)
                heavy_neighbors = _heavy_neighbors(a)
                polarized = any(
                    neighbor.elem in {"N", "O", "S", "F", "Cl", "Br", "I"}
                    for neighbor in heavy_neighbors
                )
                if hs and (not chemistry_aware or polarized):
                    carbon_donors.append((a, hs))
                if hs and _sybyl(a) in {"", "c.3"}:
                    sigma_donors.append((a, hs))
            if a.idx not in ring_atom_ids:
                if chemistry_aware and _chemistry_aware_alkyl_carbon(a):
                    alkyl_carbons.append(a)
                elif not chemistry_aware:
                    heavy = [n for n in a.neighbors if n.elem != "H"]
                    if heavy and all(n.elem == "C" for n in heavy):
                        alkyl_carbons.append(a)
        if a.elem in _HALOGENS:
            cbonded = [n for n in a.neighbors if n.elem == "C"]
            if cbonded:
                halogens.append((a, cbonded[0]))
        if a.elem in _METALS:
            metals.append(a)
        if a.elem == "S":
            sulfurs.append(a)
        if a.elem in _CHALCOGENS:
            bonded = _heavy_neighbors(a)
            if bonded:
                chalcogens.append((a, bonded[0]))

    return {
        "donors": donors,
        "carbon_donors": carbon_donors,
        "acceptors": acceptors,
        "sigma_donors": sigma_donors,
        "cations": cations,
        "anions": anions,
        "halogens": halogens,
        "alkyl": alkyl_carbons,
        "metals": metals,
        "sulfurs": sulfurs,
        "chalcogens": chalcogens,
        "rings": rings,
    }


# ===========================================================================
# Interaction detectors — ported VERBATIM except a_sele/b_sele -> a_obj/b_obj
# ===========================================================================


def _mk(
    itype,
    subtype,
    a_label,
    b_label,
    a_point,
    b_point,
    a_obj,
    b_obj,
    a_role="",
    b_role="",
    **extra,
):
    record = {
        "type": itype,
        "subtype": subtype,
        "a_label": a_label,
        "b_label": b_label,
        "a_point": a_point,
        "b_point": b_point,
        "a_obj": a_obj,
        "b_obj": b_obj,
        "a_role": a_role,
        "b_role": b_role,
    }
    record.update(extra)
    return record


def _hbond_pairs(feat_a, feat_b, itype, dist_cut, angle_cut, has_h):
    donor_key = "donors" if itype == "hbond" else "carbon_donors"
    out = []
    for donor, hs in feat_a[donor_key]:
        for acc in feat_b["acceptors"]:
            if donor.idx == acc.idx:
                continue
            d = _dist(donor.coord, acc.coord)
            if d > dist_cut:
                continue
            profile = _ACTIVE_CHEMISTRY_PROFILE.get()
            chemistry_aware = profile != "plip"
            if chemistry_aware and hs:
                cutoffs = _active_cutoffs()
                prefix = "hbond" if itype == "hbond" else "carbon_hbond"
                acceptor_bases = [
                    neighbor for neighbor in acc.neighbors if neighbor.elem != "H"
                ]
                if not acceptor_bases:
                    continue
                for hydrogen in hs:
                    hydrogen_distance = _dist(hydrogen.coord, acc.coord)
                    use_h_a = bool(cutoffs.get(f"{prefix}_h_a_required", 0.0))
                    if use_h_a and hydrogen_distance > cutoffs[
                        f"{prefix}_h_a_dist"
                    ]:
                        continue
                    donor_angle = _angle_at(
                        hydrogen.coord,
                        donor.coord,
                        acc.coord,
                    )
                    if donor_angle < angle_cut:
                        continue
                    acceptor_angle = max(
                        _angle_at(
                            acc.coord,
                            hydrogen.coord,
                            base.coord,
                        )
                        for base in acceptor_bases
                    )
                    if acceptor_angle < cutoffs[f"{prefix}_acceptor_angle"]:
                        continue
                    donor_acceptor_base_angle = max(
                        _angle_at(
                            acc.coord,
                            donor.coord,
                            base.coord,
                        )
                        for base in acceptor_bases
                    )
                    if donor_acceptor_base_angle < cutoffs.get(
                        f"{prefix}_dar_angle", 0.0
                    ):
                        continue
                    out.append(
                        _mk(
                            itype,
                            "",
                            donor.label(),
                            acc.label(),
                            donor.coord,
                            acc.coord,
                            donor,
                            acc,
                            "donor",
                            "acceptor",
                            chemistry_basis="explicit_hydrogen",
                            confidence="high",
                            hydrogen_obj=hydrogen,
                            hydrogen_acceptor_distance=hydrogen_distance,
                            donor_hydrogen_acceptor_angle=donor_angle,
                            hydrogen_acceptor_base_angle=acceptor_angle,
                            donor_acceptor_base_angle=donor_acceptor_base_angle,
                        )
                    )
                continue
            if chemistry_aware and not hs:
                if profile in {"luna", "luna_dsv"}:
                    continue
            if not chemistry_aware and has_h and hs:
                best = max(_angle_at(h.coord, donor.coord, acc.coord) for h in hs)
                if best < angle_cut:
                    continue
            chemistry_metadata = {}
            if chemistry_aware:
                chemistry_metadata = {
                    "chemistry_basis": "inferred_hydrogen",
                    "confidence": "medium",
                }
            out.append(
                _mk(
                    itype,
                    "",
                    donor.label(),
                    acc.label(),
                    donor.coord,
                    acc.coord,
                    donor,
                    acc,
                    "donor",
                    "acceptor",
                    **chemistry_metadata,
                )
            )
    return out


def detect_hbond(fa, fb, has_h):
    c = _active_cutoffs()
    res = _hbond_pairs(fa, fb, "hbond", c["hbond_dist"], c["hbond_angle"], has_h)
    res += _hbond_pairs(fb, fa, "hbond", c["hbond_dist"], c["hbond_angle"], has_h)
    return res


def detect_carbon_hbond(fa, fb, has_h):
    c = _active_cutoffs()
    res = _hbond_pairs(
        fa, fb, "carbon_hbond", c["carbon_hbond_dist"], c["carbon_hbond_angle"], has_h
    )
    res += _hbond_pairs(
        fb, fa, "carbon_hbond", c["carbon_hbond_dist"], c["carbon_hbond_angle"], has_h
    )
    return res


def detect_saltbridge(fa, fb):
    cut = _active_cutoffs()["saltbridge_dist"]
    out = []
    best_by_pair = {}
    semantic_dedup = _ACTIVE_CHEMISTRY_PROFILE.get() != "plip"
    for cats, anis in ((fa["cations"], fb["anions"]), (fb["cations"], fa["anions"])):
        for cpt, clbl, catom in cats:
            for apt, albl, aatom in anis:
                distance = _dist(cpt, apt)
                if distance > cut:
                    continue
                record = _mk(
                    "saltbridge",
                    "",
                    clbl,
                    albl,
                    cpt,
                    apt,
                    catom,
                    aatom,
                    "cation",
                    "anion",
                )
                if not semantic_dedup:
                    out.append(record)
                    continue
                key = (catom.res_tag(), aatom.res_tag())
                previous = best_by_pair.get(key)
                if previous is None or distance < previous[0]:
                    best_by_pair[key] = (distance, record)
    if semantic_dedup:
        out.extend(item[1] for item in best_by_pair.values())
    return out


def _deduplicated_charge_records(pairs, interaction_type, cutoff):
    best_by_pair = {}
    for centres_a, centres_b in pairs:
        for point_a, label_a, atom_a in centres_a:
            for point_b, label_b, atom_b in centres_b:
                distance = _dist(point_a, point_b)
                if distance > cutoff:
                    continue
                key = (atom_a.res_tag(), atom_b.res_tag())
                record = _mk(
                    interaction_type,
                    "",
                    label_a,
                    label_b,
                    point_a,
                    point_b,
                    atom_a,
                    atom_b,
                    "charge",
                    "charge",
                )
                previous = best_by_pair.get(key)
                if previous is None or distance < previous[0]:
                    best_by_pair[key] = (distance, record)
    return [item[1] for item in best_by_pair.values()]


def detect_attractive_charge(fa, fb):
    """Detect ionic attraction outside the profile's salt-bridge core."""
    profile = _ACTIVE_CHEMISTRY_PROFILE.get()
    if profile == "plip":
        return []
    cutoffs = _active_cutoffs()
    records = _deduplicated_charge_records(
        ((fa["cations"], fb["anions"]), (fb["cations"], fa["anions"])),
        "attractive_charge",
        cutoffs["attractive_charge_dist"],
    )
    if profile != "plip":
        salt_limit = cutoffs["saltbridge_dist"]
        records = [
            record
            for record in records
            if _dist(record["a_point"], record["b_point"]) > salt_limit
        ]
    return records


def detect_charge_repulsion(fa, fb):
    """Detect same-sign charge contacts exposed by both LUNA and DSV."""
    if _ACTIVE_CHEMISTRY_PROFILE.get() == "plip":
        return []
    cutoffs = _active_cutoffs()
    return _deduplicated_charge_records(
        (
            (fa["cations"], fb["cations"]),
            (fa["anions"], fb["anions"]),
        ),
        "charge_repulsion",
        cutoffs["repulsive_charge_dist"],
    )


def detect_pipi(fa, fb):
    c = _active_cutoffs()
    out = []
    for r1 in fa["rings"]:
        for r2 in fb["rings"]:
            if _dist(r1.centroid, r2.centroid) > c["pipi_dist"]:
                continue
            profile = _ACTIVE_CHEMISTRY_PROFILE.get()
            if profile != "plip":
                closest = min(
                    _dist(atom1.coord, atom2.coord)
                    for atom1 in r1.atoms
                    for atom2 in r2.atoms
                )
                closest_limit = c.get("pipi_closest_atom_dist")
                if closest_limit is not None and closest > closest_limit:
                    continue
                theta = _plane_angle(r1.normal, r2.normal)
                gamma = max(
                    _axis_angle(r2.centroid, r1.centroid, r1.normal),
                    _axis_angle(r1.centroid, r2.centroid, r2.normal),
                )
                stacked = theta <= c["pipi_stacked_theta_max"]
                if "pipi_stacked_gamma_max" in c:
                    stacked = stacked and gamma <= c["pipi_stacked_gamma_max"]
                tshaped = (
                    90.0 - theta <= c["pipi_t_theta_deviation_max"]
                )
                if "pipi_t_gamma_min" in c:
                    tshaped = tshaped and gamma >= c["pipi_t_gamma_min"]
                if stacked:
                    subtype = "sandwich"
                elif tshaped:
                    subtype = "tshaped"
                else:
                    continue
                out.append(
                    _mk(
                        "pipi",
                        subtype,
                        r1.tag,
                        r2.tag,
                        r1.centroid,
                        r2.centroid,
                        r1,
                        r2,
                        "ring",
                        "ring",
                        theta=theta,
                        gamma=gamma,
                        closest_atom_distance=closest,
                    )
                )
                continue
            offset = min(
                _proj_offset(r2.centroid, r1.centroid, r1.normal),
                _proj_offset(r1.centroid, r2.centroid, r2.normal),
            )
            if offset > c["pipi_offset"]:
                continue
            ang = _plane_angle(r1.normal, r2.normal)
            dev = c["pipi_angle_dev"]
            if ang <= dev:
                subtype = "sandwich"
            elif ang >= (90.0 - dev):
                subtype = "tshaped"
            else:
                continue
            out.append(
                _mk(
                    "pipi",
                    subtype,
                    r1.tag,
                    r2.tag,
                    r1.centroid,
                    r2.centroid,
                    r1,
                    r2,
                    "ring",
                    "ring",
                )
            )
    return out


def detect_pication(fa, fb):
    c = _active_cutoffs()
    out = []
    for rings, cats in ((fa["rings"], fb["cations"]), (fb["rings"], fa["cations"])):
        for r in rings:
            for cpt, clbl, catom in cats:
                if _dist(r.centroid, cpt) > c["pication_dist"]:
                    continue
                profile = _ACTIVE_CHEMISTRY_PROFILE.get()
                if profile == "plip":
                    if _proj_offset(cpt, r.centroid, r.normal) > c[
                        "pication_offset"
                    ]:
                        continue
                elif "pication_angle_max" in c and _axis_angle(
                    cpt, r.centroid, r.normal
                ) > c["pication_angle_max"]:
                    continue
                out.append(
                    _mk(
                        "pication",
                        "",
                        r.tag,
                        clbl,
                        r.centroid,
                        cpt,
                        r,
                        catom,
                        "ring",
                        "cation",
                    )
                )
    return out


def detect_pialkyl(fa, fb):
    cut = _active_cutoffs()["pialkyl_dist"]
    out = []
    best_by_group = {}
    semantic_profile = _ACTIVE_CHEMISTRY_PROFILE.get() != "plip"
    for rings, alks in ((fa["rings"], fb["alkyl"]), (fb["rings"], fa["alkyl"])):
        for r in rings:
            for a in alks:
                distance = _dist(r.centroid, a.coord)
                if distance > cut:
                    continue
                if semantic_profile and any(
                    _pi_sigma_geometry(r, a, hydrogen) is not None
                    for hydrogen in _h_neighbors(a)
                ):
                    continue
                record = _mk(
                    "pialkyl",
                    "",
                    r.tag,
                    a.label(),
                    r.centroid,
                    a.coord,
                    r,
                    a,
                    "ring",
                    "alkyl",
                )
                if not semantic_profile:
                    out.append(record)
                    continue
                # Use one closest contact per ring/residue pair for stable
                # residue-level counting. The export retains its atom-level
                # representative, avoiding combinatorial C--C duplicates.
                group_key = (r.tag, a.res_tag())
                previous = best_by_group.get(group_key)
                if previous is None or distance < previous[0]:
                    best_by_group[group_key] = (distance, record)
    if semantic_profile:
        out.extend(value[1] for value in best_by_group.values())
    return out


def _axis_angle(point, centre, normal):
    """Acute angle between a ring normal and a centroid-to-point vector."""
    direction = _v(point) - _v(centre)
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return 90.0
    cosine = abs(np.clip((direction / norm).dot(_v(normal)), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _pi_sigma_geometry(ring, donor, hydrogen):
    cutoffs = _active_cutoffs()
    donor_distance = _dist(ring.centroid, donor.coord)
    hydrogen_distance = _dist(ring.centroid, hydrogen.coord)
    theta = _axis_angle(hydrogen.coord, ring.centroid, ring.normal)
    donor_angle = _angle_at(hydrogen.coord, donor.coord, ring.centroid)
    if donor_distance > cutoffs["pi_sigma_carbon_dist"]:
        return None
    if bool(cutoffs.get("pi_sigma_use_h_centroid", 1.0)) and (
        hydrogen_distance > cutoffs["pi_sigma_h_centroid_dist"]
    ):
        return None
    if theta > cutoffs["pi_sigma_axis_angle"]:
        return None
    if donor_angle < cutoffs["pi_sigma_dha_angle"]:
        return None
    return donor_distance, hydrogen_distance, theta, donor_angle


def detect_pi_sigma(fa, fb):
    """Detect an axial C-H sigma bond directed toward an aromatic ring."""
    if _ACTIVE_CHEMISTRY_PROFILE.get() not in {"dsv", "luna_dsv"}:
        return []
    best_by_pair = {}
    for rings, donors in (
        (fa["rings"], fb["sigma_donors"]),
        (fb["rings"], fa["sigma_donors"]),
    ):
        for ring in rings:
            for donor, hydrogens in donors:
                for hydrogen in hydrogens:
                    geometry = _pi_sigma_geometry(ring, donor, hydrogen)
                    if geometry is None:
                        continue
                    donor_distance, hydrogen_distance, theta, donor_angle = geometry
                    record = _mk(
                        "pi_sigma",
                        "C-H/pi",
                        ring.tag,
                        donor.label(),
                        ring.centroid,
                        donor.coord,
                        ring,
                        donor,
                        "pi_orbitals",
                        "sigma_donor",
                        chemistry_basis="explicit_hydrogen",
                        confidence="medium",
                        hydrogen_obj=hydrogen,
                        hydrogen_acceptor_distance=hydrogen_distance,
                        donor_hydrogen_acceptor_angle=donor_angle,
                        theta=theta,
                    )
                    key = (ring.tag, donor.idx)
                    previous = best_by_pair.get(key)
                    if previous is None or hydrogen_distance < previous[0]:
                        best_by_pair[key] = (hydrogen_distance, record)
    return [value[1] for value in best_by_pair.values()]


def detect_pi_donor_hbond(fa, fb):
    """Detect an N/O-H donor directed toward an aromatic pi system."""
    if _ACTIVE_CHEMISTRY_PROFILE.get() not in {"dsv", "luna_dsv"}:
        return []
    cutoffs = _active_cutoffs()
    best_by_pair = {}
    for rings, donors in (
        (fa["rings"], fb["donors"]),
        (fb["rings"], fa["donors"]),
    ):
        for ring in rings:
            for donor, hydrogens in donors:
                for hydrogen in hydrogens:
                    donor_distance = _dist(ring.centroid, donor.coord)
                    hydrogen_distance = _dist(ring.centroid, hydrogen.coord)
                    theta = _axis_angle(hydrogen.coord, ring.centroid, ring.normal)
                    donor_angle = _angle_at(hydrogen.coord, donor.coord, ring.centroid)
                    if donor_distance > cutoffs["pi_donor_dist"]:
                        continue
                    if bool(cutoffs.get("pi_donor_use_h_centroid", 1.0)) and (
                        hydrogen_distance > cutoffs["pi_donor_h_centroid_dist"]
                    ):
                        continue
                    if theta > cutoffs["pi_donor_axis_angle"]:
                        continue
                    if donor_angle < cutoffs["pi_donor_dha_angle"]:
                        continue
                    record = _mk(
                        "pi_donor_hbond",
                        "X-H/pi",
                        ring.tag,
                        donor.label(),
                        ring.centroid,
                        donor.coord,
                        ring,
                        donor,
                        "pi_acceptor",
                        "donor",
                        chemistry_basis="explicit_hydrogen",
                        confidence="medium",
                        hydrogen_obj=hydrogen,
                        hydrogen_acceptor_distance=hydrogen_distance,
                        donor_hydrogen_acceptor_angle=donor_angle,
                        theta=theta,
                    )
                    key = (ring.tag, donor.idx)
                    previous = best_by_pair.get(key)
                    if previous is None or hydrogen_distance < previous[0]:
                        best_by_pair[key] = (hydrogen_distance, record)
    return [value[1] for value in best_by_pair.values()]


def detect_alkyl(fa, fb):
    cut = _active_cutoffs()["alkyl_dist"]
    out = []
    best_by_pair = {}
    semantic_dedup = _ACTIVE_CHEMISTRY_PROFILE.get() != "plip"
    for a in fa["alkyl"]:
        for b in fb["alkyl"]:
            distance = _dist(a.coord, b.coord)
            if distance > cut:
                continue
            record = _mk(
                    "alkyl",
                    "",
                    a.label(),
                    b.label(),
                    a.coord,
                    b.coord,
                    a,
                    b,
                    "alkyl",
                    "alkyl",
                )
            if not semantic_dedup:
                out.append(record)
                continue
            key = (a.res_tag(), b.res_tag())
            previous = best_by_pair.get(key)
            if previous is None or distance < previous[0]:
                best_by_pair[key] = (distance, record)
    if semantic_dedup:
        out.extend(item[1] for item in best_by_pair.values())
    return out


def detect_halogen(fa, fb):
    c = _active_cutoffs()
    out = []
    for hals, accs in (
        (fa["halogens"], fb["acceptors"]),
        (fb["halogens"], fa["acceptors"]),
    ):
        for x, cbonded in hals:
            for acc in accs:
                distance = _dist(x.coord, acc.coord)
                profile = _ACTIVE_CHEMISTRY_PROFILE.get()
                if profile == "plip":
                    if distance > c["halogen_dist"]:
                        continue
                    if _angle_at(x.coord, cbonded.coord, acc.coord) < c[
                        "halogen_angle"
                    ]:
                        continue
                else:
                    distance_limits = []
                    if profile in {"luna", "luna_dsv"}:
                        distance_limits.append(c["halogen_dist"])
                    if profile in {"dsv", "luna_dsv"}:
                        if x.elem == "F":
                            distance_limits.append(c["halogen_f_dist"])
                        else:
                            vdw_limit = c["halogen_vdw_fraction"] * (
                                _VDW_RADII.get(x.elem, 1.70)
                                + _VDW_RADII.get(acc.elem, 1.70)
                            )
                            distance_limits.append(vdw_limit)
                    if distance > min(distance_limits):
                        continue
                    donor_angle = _angle_at(x.coord, cbonded.coord, acc.coord)
                    if donor_angle < c["halogen_donor_angle_min"]:
                        continue
                    bases = _heavy_neighbors(acc)
                    if not bases:
                        continue
                    acceptor_angle = max(
                        _angle_at(acc.coord, x.coord, base.coord)
                        for base in bases
                    )
                    if acceptor_angle < c["halogen_acceptor_angle_min"]:
                        continue
                out.append(
                    _mk(
                        "halogen",
                        "",
                        x.label(),
                        acc.label(),
                        x.coord,
                        acc.coord,
                        x,
                        acc,
                        "halogen",
                        "acceptor",
                    )
                )
    return out


def detect_metal(fa, fb):
    cut = _active_cutoffs()["metal_dist"]
    out = []
    for metals, accs in (
        (fa["metals"], fb["acceptors"]),
        (fb["metals"], fa["acceptors"]),
    ):
        for m in metals:
            for acc in accs:
                if _dist(m.coord, acc.coord) <= cut:
                    out.append(
                        _mk(
                            "metal",
                            "",
                            m.label(),
                            acc.label(),
                            m.coord,
                            acc.coord,
                            m,
                            acc,
                            "metal",
                            "acceptor",
                        )
                    )
    return out


def detect_pi_sulfur(fa, fb):
    cutoffs = _active_cutoffs()
    cut = cutoffs["pi_sulfur_dist"]
    out = []
    profile = _ACTIVE_CHEMISTRY_PROFILE.get()
    strict_aromatic = profile != "plip"
    for rings, sulfs in ((fa["rings"], fb["sulfurs"]), (fb["rings"], fa["sulfurs"])):
        for r in rings:
            if strict_aromatic and not _ring_has_aromatic_evidence(r):
                continue
            for s in sulfs:
                distance = _dist(r.centroid, s.coord)
                subtype = ""
                if profile in {"dsv", "luna_dsv"}:
                    axis_angle = _axis_angle(s.coord, r.centroid, r.normal)
                    face_limit = min(
                        cut,
                        cutoffs.get("pi_sulfur_face_dist", cut),
                    )
                    face = (
                        distance <= face_limit
                        and axis_angle
                        <= cutoffs.get("pi_sulfur_face_angle_max", 25.0)
                    )
                    edge = (
                        distance <= cut
                        and axis_angle
                        >= cutoffs.get("pi_sulfur_edge_angle_min", 70.0)
                    )
                    if face:
                        subtype = "face-on"
                    elif edge:
                        subtype = "edge-on"
                    else:
                        continue
                elif profile == "luna":
                    if distance > cut or _axis_angle(
                        s.coord, r.centroid, r.normal
                    ) > cutoffs.get("chalcogen_displacement_angle_max", 60.0):
                        continue
                    subtype = "chalcogen-pi"
                elif distance > cut:
                    continue
                if distance <= cut:
                    out.append(
                        _mk(
                            "pi_sulfur",
                            subtype,
                            r.tag,
                            s.label(),
                            r.centroid,
                            s.coord,
                            r,
                            s,
                            "ring",
                            "sulfur",
                        )
                    )
    return out


def detect_chalcogen(fa, fb):
    """Detect conservative R-Y...A-N chalcogen bonds (Y = S, Se or Te)."""
    cutoffs = _active_cutoffs()
    profile = _ACTIVE_CHEMISTRY_PROFILE.get()
    if profile == "plip":
        return []
    best_by_pair = {}
    for donors, acceptors in (
        (fa["chalcogens"], fb["acceptors"]),
        (fb["chalcogens"], fa["acceptors"]),
    ):
        for chalcogen, bonded in donors:
            for acceptor in acceptors:
                bases = _heavy_neighbors(acceptor)
                if not bases:
                    continue
                distance = _dist(chalcogen.coord, acceptor.coord)
                limits = []
                if profile in {"luna", "luna_dsv"}:
                    limits.append(cutoffs["chalcogen_dist"])
                if profile in {"dsv", "luna_dsv"}:
                    limits.append(
                        cutoffs["chalcogen_vdw_fraction"]
                        * (
                            _VDW_RADII.get(chalcogen.elem, 1.80)
                            + _VDW_RADII.get(acceptor.elem, 1.70)
                        )
                    )
                if not limits or distance > min(limits):
                    continue
                donor_angle = _angle_at(
                    chalcogen.coord,
                    bonded.coord,
                    acceptor.coord,
                )
                if donor_angle < cutoffs["chalcogen_donor_angle_min"]:
                    continue
                acceptor_angle = max(
                    _angle_at(acceptor.coord, chalcogen.coord, base.coord)
                    for base in bases
                )
                if acceptor_angle < cutoffs["chalcogen_acceptor_angle_min"]:
                    continue
                record = _mk(
                    "chalcogen",
                    chalcogen.elem,
                    chalcogen.label(),
                    acceptor.label(),
                    chalcogen.coord,
                    acceptor.coord,
                    chalcogen,
                    acceptor,
                    "chalcogen_donor",
                    "acceptor",
                    donor_angle=donor_angle,
                    acceptor_angle=acceptor_angle,
                )
                key = (chalcogen.res_tag(), acceptor.res_tag())
                previous = best_by_pair.get(key)
                if previous is None or distance < previous[0]:
                    best_by_pair[key] = (distance, record)
    return [item[1] for item in best_by_pair.values()]


def detect_pi_anion(fa, fb):
    c = _active_cutoffs()
    out = []
    for rings, anis in ((fa["rings"], fb["anions"]), (fb["rings"], fa["anions"])):
        for r in rings:
            for apt, albl, aatom in anis:
                if _dist(r.centroid, apt) > c["pi_anion_dist"]:
                    continue
                if _proj_offset(apt, r.centroid, r.normal) > c["pi_anion_offset"]:
                    continue
                out.append(
                    _mk(
                        "pi_anion",
                        "",
                        r.tag,
                        albl,
                        r.centroid,
                        apt,
                        r,
                        aatom,
                        "ring",
                        "anion",
                    )
                )
    return out


def detect_pi_lone_pair(fa, fb):
    """Detect a lone-pair atom aligned over the face of an aromatic ring."""
    if _ACTIVE_CHEMISTRY_PROFILE.get() not in {"dsv", "luna_dsv"}:
        return []
    cutoffs = _active_cutoffs()
    out = []
    for acceptors, rings in (
        (fa["acceptors"], fb["rings"]),
        (fb["acceptors"], fa["rings"]),
    ):
        for acceptor in acceptors:
            for ring in rings:
                if not _ring_has_aromatic_evidence(ring):
                    continue
                distance = _dist(acceptor.coord, ring.centroid)
                if distance > cutoffs["pi_lone_pair_dist"]:
                    continue
                direction = _v(acceptor.coord) - _v(ring.centroid)
                norm = np.linalg.norm(direction)
                if norm < 1e-6:
                    continue
                cosine = abs(
                    np.clip((direction / norm).dot(_v(ring.normal)), -1.0, 1.0)
                )
                theta = float(np.degrees(np.arccos(cosine)))
                if theta > cutoffs["pi_lone_pair_angle"]:
                    continue
                out.append(
                    _mk(
                        "pi_lone_pair",
                        "lone-pair/pi",
                        acceptor.label(),
                        ring.tag,
                        acceptor.coord,
                        ring.centroid,
                        acceptor,
                        ring,
                        "lone_pair",
                        "pi_orbitals",
                        theta=theta,
                    )
                )
    return out


def detect_water_bridge(fa, fb, waters):
    """Return one semantic receptor-water-ligand record per bridge."""
    c = _active_cutoffs()

    def _partners(feat):
        donors = {atom.idx: atom for atom, _hs in feat["donors"]}
        acceptors = {atom.idx: atom for atom in feat["acceptors"]}
        ordered_ids = tuple(donors) + tuple(
            index for index in acceptors if index not in donors
        )
        return [
            (
                donors[index] if index in donors else acceptors[index],
                (
                    "donor_acceptor"
                    if index in donors and index in acceptors
                    else "donor"
                    if index in donors
                    else "acceptor"
                ),
            )
            for index in ordered_ids
        ]

    pa, pb = _partners(fa), _partners(fb)
    lo, hi = c["water_bridge_min"], c["water_bridge_max"]
    amin, amax = c["water_bridge_angle_min"], c["water_bridge_angle_max"]
    out = []
    for w in waters:
        near_a = [p for p in pa if lo <= _dist(w.coord, p[0].coord) <= hi]
        near_b = [p for p in pb if lo <= _dist(w.coord, p[0].coord) <= hi]
        for pai, role_a in near_a:
            for pbi, role_b in near_b:
                ang = _angle_at(w.coord, pai.coord, pbi.coord)
                if not (amin <= ang <= amax):
                    continue
                out.append(
                    _mk(
                        "water_bridge",
                        "",
                        pai.label(),
                        pbi.label(),
                        pai.coord,
                        pbi.coord,
                        pai,
                        pbi,
                        role_a,
                        role_b,
                        water_obj=w,
                        receptor_water_distance=_dist(w.coord, pai.coord),
                        ligand_water_distance=_dist(w.coord, pbi.coord),
                        water_angle=ang,
                    )
                )
    return out


DETECTORS = {
    "hbond": lambda fa, fb, h: detect_hbond(fa, fb, h),
    "carbon_hbond": lambda fa, fb, h: detect_carbon_hbond(fa, fb, h),
    "saltbridge": lambda fa, fb, h: detect_saltbridge(fa, fb),
    "attractive_charge": lambda fa, fb, h: detect_attractive_charge(fa, fb),
    "charge_repulsion": lambda fa, fb, h: detect_charge_repulsion(fa, fb),
    "pipi": lambda fa, fb, h: detect_pipi(fa, fb),
    "pication": lambda fa, fb, h: detect_pication(fa, fb),
    "pialkyl": lambda fa, fb, h: detect_pialkyl(fa, fb),
    "pi_sigma": lambda fa, fb, h: detect_pi_sigma(fa, fb),
    "alkyl": lambda fa, fb, h: detect_alkyl(fa, fb),
    "halogen": lambda fa, fb, h: detect_halogen(fa, fb),
    "metal": lambda fa, fb, h: detect_metal(fa, fb),
    "pi_sulfur": lambda fa, fb, h: detect_pi_sulfur(fa, fb),
    "pi_anion": lambda fa, fb, h: detect_pi_anion(fa, fb),
    "pi_donor_hbond": lambda fa, fb, h: detect_pi_donor_hbond(fa, fb),
    "pi_lone_pair": lambda fa, fb, h: detect_pi_lone_pair(fa, fb),
    "chalcogen": lambda fa, fb, h: detect_chalcogen(fa, fb),
    # water_bridge handled separately (needs the water list)
}


# ===========================================================================
# High-level driver + endpoint metadata helpers
# ===========================================================================


def endpoint_side(obj):
    """'receptor' | 'ligand' | 'water' for an interaction endpoint (Atom/Ring)."""
    if isinstance(obj, Ring):
        return obj.atoms[0].side
    return obj.side


def endpoint_resid(obj):
    """Residue tag (resn+resi+chain) of an endpoint."""
    a = obj.atoms[0] if isinstance(obj, Ring) else obj
    return a.res_tag()


def endpoint_name(obj):
    """Atom name, or 'ring'/'ring2' for a ring endpoint."""
    if isinstance(obj, Ring):
        return obj.tag.rsplit("_", 1)[-1]
    return obj.name


def compute_interactions(
    receptor_atoms,
    ligand_atoms,
    waters=None,
    types=None,
    cutoffs=None,
    chemistry_profile="plip",
):
    """Detect all requested interactions between receptor and ligand.

    Atoms must already have `.side` set ('receptor'/'ligand'/'water'). Returns a
    list of interaction dicts, each with an added 'dist' (endpoint separation).
    """
    chemistry_profile = str(chemistry_profile).strip().lower()
    if chemistry_profile not in HBOND_PRESETS:
        raise ValueError("Unknown chemistry profile: %s" % chemistry_profile)
    effective_cutoffs = dict(
        cutoffs if cutoffs is not None else cutoffs_for_preset(chemistry_profile)
    )
    cutoff_token = _ACTIVE_CUTOFFS.set(MappingProxyType(effective_cutoffs))
    chemistry_token = _ACTIVE_CHEMISTRY_PROFILE.set(chemistry_profile)
    try:
        waters = waters or []
        req = (
            list(types)
            if types
            else list(default_types_for_profile(chemistry_profile))
        )
        # ``has_h`` preserves the legacy PLIP behavior and controls explicit
        # carbon-donor geometry. Chemistry-aware donor fallback is deliberately
        # assessed per molecular side: DSV may infer donors on a hydrogen-free
        # partner without discarding explicit geometry on the protonated side,
        # while LUNA and the conservative hybrid always require bonded H atoms.
        has_h = any(a.elem == "H" for a in (*receptor_atoms, *ligand_atoms))

        feat_r = classify(
            receptor_atoms,
            _build_rings(receptor_atoms),
            has_h,
            chemistry_profile=chemistry_profile,
        )
        feat_l = classify(
            ligand_atoms,
            _build_rings(ligand_atoms),
            has_h,
            chemistry_profile=chemistry_profile,
        )

        inters = []
        for itype in req:
            if itype == "water_bridge":
                inters.extend(detect_water_bridge(feat_r, feat_l, waters))
            elif itype in DETECTORS:
                inters.extend(DETECTORS[itype](feat_r, feat_l, has_h))
        for it in inters:
            it["dist"] = _dist(it["a_point"], it["b_point"])
        return inters
    finally:
        _ACTIVE_CHEMISTRY_PROFILE.reset(chemistry_token)
        _ACTIVE_CUTOFFS.reset(cutoff_token)
