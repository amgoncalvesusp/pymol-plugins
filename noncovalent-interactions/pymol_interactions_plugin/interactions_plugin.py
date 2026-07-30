"""
interactions_plugin.py — Non-covalent interaction detector/visualizer for PyMOL
===============================================================================

Detects and draws non-covalent intermolecular interactions (beyond PyMOL's
native H-bond dashes) as individual, colour-coded dashed-line objects for
docking-pose and MD-trajectory analysis.

INSTALL
-------
  * Plugin Manager: Plugin > Manage Plugins > Install... > pick this file.
  * Or from the PyMOL command line:   run /path/to/interactions_plugin.py

USAGE (PyMOL command line)
--------------------------
  detect_interactions sel1, sel2 [, types [, state [, disable_native_hbond
                                   [, group_name [, label ]]]]]

  detect_interactions polymer, organic
  detect_interactions polymer, auto                      # auto-detect ligand
  detect_interactions polymer, resn LIG, types=hbond pipi saltbridge
  detect_interactions polymer, organic, state=5          # one MD frame
  detect_interactions polymer, organic, disable_native_hbond=1
  detect_interactions polymer, organic, show_residues=1  # residues as sticks
  detect_interactions polymer, organic, engine=ds         # DS-style cutoffs

COMMANDS
  detect_interactions      detect + draw for one state (main command)
  interactions_set_engine  switch detection engine: 'ds' (default) or 'plip'
                           ('ds' is the DockLens parity contract); also
                           a radio-button toggle in interactions_gui
  interactions_occupancy   persistence (%) of each interaction over an MD
                           trajectory: loops states, prints/CSV a ranked table
  interactions_export_csv  dump one state's interactions to a CSV file
  interactions_figure_preset  publication display settings (+ optional PNG)
  interactions_set_appearance  dash thickness / length-gap scale / label size
  interactions_visibility  show | hide | clear the dashes, residues and legend
  interactions_set_cutoff  edit one CUTOFFS entry at runtime (or 'reset')
  show_interaction_legend  colour -> type table (onscreen=1 draws a 3D legend)
  interactions_gui         open the Qt dialog (also under Plugin menu)

  sel1  receptor side  (default 'polymer')
  sel2  ligand side    (default 'organic'; 'auto' -> organic else non-polymer)
        -- keep sel1/sel2 as DISTINCT groups
  types space/comma list, or 'all'. Valid (15):
        hbond carbon_hbond saltbridge pipi pication pialkyl alkyl halogen
        metal water_bridge pi_sulfur pi_anion pi_sigma pi_donor_hbond pi_lone_pair
  state  model state used for coordinates (1 = first; for a trajectory pass the
         frame number and re-run to recompute per frame).
  disable_native_hbond  1 => hide PyMOL's own polar-contact dashes to avoid
         duplicate H-bond lines (only hides objects named like *polar_contact*
         / *_hbond* / *hbonds*; it never deletes them).
  group_name  panel group that collects every interaction object.
  label  1 => keep the numeric distance label on each dash.
  show_residues  1 => show interacting residues as sticks in <group>_residues.

Each interaction becomes one named object, e.g.:
  LYS68_NZ--LIG_O1_saltbridge   PHE45_ring--LIG_ring_pipi   LIG_Cl1--ASP90_carboxyl_halogen

COLOURS  (Okabe-Ito, colour-blind-safe; see show_interaction_legend)
-------
  hbond          Sky-blue        #56B4E9
  carbon_hbond   Bluish-green    #009E73
  saltbridge     Vermillion      #D55E00
  pipi           Reddish-purple  #CC79A7  (sandwich = long dash, T-shaped = short dash)
  pication       Yellow          #F0E442
  pialkyl        Orange          #E69F00
  alkyl          Blue            #0072B2
  halogen        Black           #000000
  Extended (>8 types, drawn DOTTED with a reused hue to stay distinct):
  metal          Vermillion  |  water_bridge  Sky-blue
  pi_sulfur      Reddish-purple  |  pi_anion   Yellow

The default mode uses the bundled, reviewed DockLens interaction core and
applies the same ``dsv`` chemistry plus ``ds_like`` analysis filter before
drawing, counting, occupancy or CSV export. PLIP remains a separate mode.
Ring perception is topology-based (bond-graph cycles +
planarity heuristic) — no RDKit dependency.
"""

from __future__ import print_function

import importlib.util
import hashlib
import inspect
import math
import os
import re
import types

import numpy as np
from pymol import cmd

_EXPECTED_DOCKLENS_CORE_SHA256 = (
    "dfdee432587c26f5cbd1025ecd110d160966b805083100aec2832970ae99af1b"
)
_EXPECTED_ANALYSIS_PROFILE_SHA256 = (
    "8f33342682f5ae5568ae7acc728f04584695d26facba22ff271cb8bf80341823"
)


def _bundled_path(filename):
    """Locate bundled modules even when PyMOL's ``run`` rewrites __file__."""
    script_hint = inspect.currentframe().f_code.co_filename
    candidates = []
    if os.path.isabs(script_hint):
        candidates.append(os.path.dirname(script_hint))
    module_file = globals().get("__file__", "")
    if module_file and os.path.isabs(module_file):
        candidates.append(os.path.dirname(module_file))
    for base in candidates:
        path = os.path.join(base, "pymol_interactions_plugin", filename)
        if os.path.isfile(path):
            return path
    raise ImportError(
        "Could not locate bundled %s. Install pymol_interactions_plugin.zip "
        "with PyMOL Plugin Manager." % filename
    )


def _verify_reviewed_source(path, expected_sha256):
    with open(path, "rb") as source_file:
        actual = hashlib.sha256(source_file.read()).hexdigest()
    if actual != expected_sha256:
        raise ImportError(
            "Bundled DockLens module failed integrity verification: %s" % path
        )
    return path


try:
    from . import docklens_core as _docklens_core
except (ImportError, ValueError):
    _core_path = _verify_reviewed_source(
        _bundled_path("docklens_core.py"),
        _EXPECTED_DOCKLENS_CORE_SHA256,
    )
    _core_spec = importlib.util.spec_from_file_location(
        "pymol_interactions_docklens_core",
        _core_path,
    )
    if _core_spec is None or _core_spec.loader is None:
        raise ImportError("Could not load the bundled DockLens interaction core")
    _docklens_core = importlib.util.module_from_spec(_core_spec)
    _core_spec.loader.exec_module(_docklens_core)

try:
    from . import docklens_analysis_profiles as _docklens_analysis_profiles
except (ImportError, ValueError):
    _profiles_path = _verify_reviewed_source(
        _bundled_path("docklens_analysis_profiles.py"),
        _EXPECTED_ANALYSIS_PROFILE_SHA256,
    )
    _profiles_spec = importlib.util.spec_from_file_location(
        "pymol_interactions_docklens_analysis_profiles",
        _profiles_path,
    )
    if _profiles_spec is None or _profiles_spec.loader is None:
        raise ImportError(
            "Could not load the bundled DockLens analysis-profile helpers"
        )
    _docklens_analysis_profiles = importlib.util.module_from_spec(_profiles_spec)
    _profiles_spec.loader.exec_module(_docklens_analysis_profiles)


DSV_PARITY_CONTRACT = "docklens-dsv-2026.07"
PLUGIN_VERSION = "0.4.1"


def _module_source_sha256(module):
    source = inspect.getsource(module).replace("\r\n", "\n")
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _parity_sources_are_reviewed():
    try:
        return (
            _module_source_sha256(_docklens_core)
            == _EXPECTED_DOCKLENS_CORE_SHA256
            and _module_source_sha256(_docklens_analysis_profiles)
            == _EXPECTED_ANALYSIS_PROFILE_SHA256
        )
    except (OSError, TypeError):
        return False


# ===========================================================================
# Colour palette (Okabe-Ito) and per-type dash style
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
    "pipi": ("reddishpurple", "pi-pi stacking (sandwich/T-shaped)"),
    "pication": ("yellow", "pi-cation"),
    "pialkyl": ("orange", "pi-alkyl"),
    "pi_sigma": ("reddishpurple", "Pi-sigma C-H/pi (dotted)"),
    "alkyl": ("blue", "Alkyl-alkyl (hydrophobic)"),
    "halogen": ("black", "Halogen bond"),
    # --- extended set (>8 types): colour is reused, drawn with a DOTTED dash so
    # it stays distinguishable from the 8 core types that share the same hue. ---
    "metal": ("vermillion", "Metal coordination (dotted)"),
    "water_bridge": ("skyblue", "Water-mediated H-bond (dotted)"),
    "pi_sulfur": ("reddishpurple", "pi-sulfur (dotted)"),
    "pi_anion": ("yellow", "pi-anion (dotted)"),
    "pi_donor_hbond": ("bluishgreen", "Pi-donor hydrogen bond (dotted)"),
    "pi_lone_pair": ("skyblue", "Pi-lone-pair (dotted)"),
}
# Use DockLens' canonical ordering, colours and labels verbatim.
INTERACTION_COLORS = dict(_docklens_core.INTERACTION_COLORS)

# Extended types are rendered dotted to disambiguate the reused hue.
_EXTENDED_TYPES = {
    "metal", "water_bridge", "pi_sulfur", "pi_anion", "pi_sigma",
    "pi_donor_hbond", "pi_lone_pair",
}


def _color_name(itype):
    """PyMOL colour name registered for an interaction type."""
    return "ii_" + itype


def _register_colors():
    """Register one PyMOL colour per interaction type (idempotent)."""
    for itype, (okabe_key, _label) in INTERACTION_COLORS.items():
        rgb01 = [c / 255.0 for c in _OKABE_ITO[okabe_key]]
        cmd.set_color(_color_name(itype), rgb01)


# Dash style per type: (dash_length, dash_gap, dash_radius).
# pipi is overridden per-subtype at draw time (sandwich vs T-shaped).
_DASH_STYLE = {
    "hbond": (0.35, 0.35, 0.06),
    "carbon_hbond": (0.20, 0.45, 0.06),
    "saltbridge": (0.50, 0.25, 0.08),
    "pipi": (0.50, 0.30, 0.07),  # sandwich default
    "pication": (0.40, 0.30, 0.07),
    "pialkyl": (0.25, 0.40, 0.06),
    "alkyl": (0.20, 0.50, 0.05),
    "halogen": (0.45, 0.30, 0.07),
    # extended types: dotted (tiny dash, wide gap)
    "metal": (0.05, 0.35, 0.10),
    "water_bridge": (0.08, 0.32, 0.05),
    "pi_sulfur": (0.10, 0.32, 0.07),
    "pi_anion": (0.08, 0.32, 0.07),
    "pi_sigma": (0.10, 0.32, 0.07),
    "pi_donor_hbond": (0.10, 0.32, 0.07),
    "pi_lone_pair": (0.08, 0.32, 0.07),
}
_PIPI_SANDWICH_DASH = (0.60, 0.20, 0.08)  # long dash
_PIPI_TSHAPED_DASH = (0.15, 0.45, 0.08)  # short dash


# ===========================================================================
# Geometric cutoffs  (edit here to override)  — two selectable engines
# ===========================================================================
#
# Sources:
#   PLIP    = Salentin et al. 2015, PLIP config.py defaults.
#   Steiner = Steiner, Angew. Chem. Int. Ed. 2002 (weak H-bonds).
#   DS      = BIOVIA Discovery Studio Visualizer "Non-bond Interaction
#             Monitor" defaults (proprietary; values below are the ranges
#             commonly reported in DS documentation/tutorials and published
#             docking studies that cite them — approximate, flagged
#             UNCERTAIN where no single agreed number exists). The DS engine
#             is generally more permissive on distance and uses a looser
#             donor-H...acceptor angle floor than PLIP.
#
# Same detection code (detect_hbond, detect_pipi, ...) runs under both
# profiles; only the numeric thresholds in CUTOFFS change. water_bridge,
# pi_sulfur and pi_anion are not formally defined by DS, so the "ds" profile
# reuses the PLIP/literature values for those three.
def _cutoff_profile(engine):
    preset = "dsv" if engine == "ds" else "plip"
    values = dict(_docklens_core.cutoffs_for_preset(preset))
    if engine == "plip":
        # These channels are intentionally retained in the editable table even
        # though DockLens calibrates them only in the DSV branch.
        values.update(
            {
                "pi_sigma_carbon_dist": 4.5,
                "pi_sigma_h_centroid_dist": 4.3,
                "pi_sigma_axis_angle": 40.0,
                "pi_sigma_dha_angle": 160.0,
                "pi_donor_dist": 5.2,
                "pi_donor_h_centroid_dist": 4.1,
                "pi_donor_axis_angle": 45.0,
                "pi_donor_dha_angle": 145.0,
                "pi_lone_pair_dist": 3.5,
                "pi_lone_pair_angle": 30.0,
            }
        )
    return values


CUTOFF_PROFILES = {engine: _cutoff_profile(engine) for engine in ("plip", "ds")}
DETECTION_ENGINES = list(CUTOFF_PROFILES.keys())  # ["plip", "ds"]

# Active cutoff table (mutated in place by interactions_set_engine /
# interactions_set_cutoff so every detector, which reads the CUTOFFS global
# directly, immediately sees the change).
CUTOFFS = dict(CUTOFF_PROFILES["ds"])
_active_engine = ["ds"]
_last_parity_diagnostics = []
_parity_customized = [False]
_parity_source_integrity = [_parity_sources_are_reviewed()]

MAX_ATOM_PAIRS = 5_000_000
MAX_COMPUTE_COST_PER_FRAME = 10_000_000
MAX_OCCUPANCY_COMPUTE_COST = 50_000_000
MAX_DRAWN_INTERACTIONS = 2_000
MAX_OCCUPANCY_FRAMES = 10_000
_last_compute_cost = [0]
_GROUP_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")
_RESERVED_GROUP_NAMES = frozenset(
    {"all", "everything", "none", "enabled", "disabled", "visible"}
)


def _parity_is_active():
    return (
        _active_engine[0] == "ds"
        and not _parity_customized[0]
        and _parity_source_integrity[0]
        and not _last_parity_diagnostics
    )


def _estimate_compute_cost(n_receptor, n_ligand, n_waters, include_water):
    atom_pairs = n_receptor * n_ligand
    if not include_water:
        return atom_pairs
    return atom_pairs + n_waters * (
        n_receptor + n_ligand + atom_pairs
    )


# Pristine copy of the *active engine's* defaults (for the GUI "Reset" in the
# cutoff editor). Refreshed by interactions_set_engine on every switch.
_CUTOFF_DEFAULTS = dict(CUTOFFS)

VALID_TYPES = list(INTERACTION_COLORS.keys())

# Planarity tolerance for aromatic-ring perception (max atom deviation from the
# best-fit plane, in Angstrom). Aromatic rings are near-flat (<~0.1 A); a
# cyclohexane chair puckers ~0.25 A, so keep the cut below that to reject sp3
# rings while tolerating thermal/MD distortion of true aromatics.
_RING_PLANARITY_TOL = 0.15
_RING_ELEMENTS = {"C", "N", "O", "S"}  # aromatic-capable ring atoms


# ===========================================================================
# Small vector helpers
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
    return vh[2]  # smallest-singular-value direction


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
    """Lateral offset of `point` from the axis through `plane_point` along
    `normal` (offset of its projection onto the ring plane)."""
    d = _v(point) - _v(plane_point)
    along = d.dot(_v(normal))
    perp = d - along * _v(normal)
    return float(np.linalg.norm(perp))


def _axis_angle(point, centre, normal):
    """Acute angle between a ring normal and its centroid-to-point vector."""
    direction = _v(point) - _v(centre)
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return 90.0
    cosine = abs(np.clip((direction / norm).dot(_v(normal)), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


# ===========================================================================
# Model loading + topology
# ===========================================================================


class Atom(object):
    """Lightweight atom wrapper around a chempy atom + its state coordinate."""

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
        "model",
        "pymol_index",
    )

    def __init__(self, idx, catom):
        self.idx = idx
        self.elem = (catom.symbol or catom.name[:1]).strip().capitalize()
        self.name = catom.name.strip()
        self.resn = catom.resn.strip()
        self.resi = catom.resi.strip()
        self.chain = catom.chain.strip()
        self.segi = catom.segi.strip()
        self.coord = _v(catom.coord)
        try:
            self.fcharge = int(catom.formal_charge)
        except Exception:
            self.fcharge = 0
        self.neighbors = []  # list of Atom (filled from bonds)
        self.serial = getattr(catom, "id", None)
        if self.serial is None:
            self.serial = idx + 1
        self.subst_id = None
        self.side = None
        raw_type = getattr(catom, "text_type", "")
        if not raw_type:
            fallback_type = getattr(catom, "type", "")
            raw_type = fallback_type if isinstance(fallback_type, str) else ""
        self.sybyl_type = str(raw_type or "").strip()
        if self.sybyl_type and self.sybyl_type != "??":
            # DockLens' MOL2/PDBQT parsers treat their charge column as partial
            # charge only. PyMOL may infer formal charges from the same types;
            # ignoring that inference keeps the two pipelines identical.
            self.fcharge = 0
        try:
            value = getattr(catom, "partial_charge", None)
            self.partial_charge = float(value) if value is not None else None
        except (TypeError, ValueError):
            self.partial_charge = None
        self.bond_orders = {}
        self.model = str(getattr(catom, "model", "") or "")
        try:
            self.pymol_index = int(getattr(catom, "index"))
        except (AttributeError, TypeError, ValueError):
            self.pymol_index = None

    def res_tag(self):
        chain = self.chain if self.chain else "_"
        return "%s%s%s" % (self.resn, self.resi, ("" if chain == "_" else chain))

    def label(self):
        return "%s_%s" % (self.res_tag(), self.name)

    def res_sele(self):
        """PyMOL selection string matching this atom's whole residue."""
        model_name = getattr(self, "model", "")
        pymol_index = getattr(self, "pymol_index", None)
        if model_name and pymol_index is not None:
            model = model_name.replace("\\", "\\\\").replace('"', '\\"')
            return '(byres (model "%s" and index %d))' % (
                model,
                pymol_index,
            )
        parts = ["resn %s" % self.resn, "resi \\%s" % self.resi]
        if self.chain:
            parts.append("chain %s" % self.chain)
        if self.segi:
            parts.append("segi %s" % self.segi)
        return "(" + " and ".join(parts) + ")"


def _load_atoms(selection, state, index_offset=0):
    """Return (atoms, has_hydrogen) for a selection at a given state."""
    model = cmd.get_model(selection, state=state)
    atoms = [Atom(index_offset + i, ca) for i, ca in enumerate(model.atom)]
    for bond in model.bond:
        i, j = bond.index
        atoms[i].neighbors.append(atoms[j])
        atoms[j].neighbors.append(atoms[i])
        raw_order = getattr(bond, "order", 1)
        try:
            numeric_order = float(raw_order)
            if numeric_order == 4.0:
                order = "ar"
            elif numeric_order.is_integer():
                order = str(int(numeric_order))
            else:
                order = str(numeric_order)
        except (TypeError, ValueError):
            order = str(raw_order or "1").strip().lower()
        atoms[i].bond_orders[atoms[j].idx] = order
        atoms[j].bond_orders[atoms[i].idx] = order
    has_h = any(a.elem == "H" for a in atoms)
    return atoms, has_h


def _find_rings(atoms):
    """Topology-based aromatic-ring perception.

    Finds all 5- and 6-membered cycles in the bond graph, then keeps those that
    are (a) built only from aromatic-capable elements (C/N/O/S) and (b) planar
    within _RING_PLANARITY_TOL. Returns a list of atom lists (one per ring).
    """
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
    """Aromatic ring feature: centroid, plane normal, label."""

    __slots__ = ("atoms", "centroid", "normal", "tag")

    def __init__(self, members, tag):
        self.atoms = members
        coords = [a.coord for a in members]
        self.centroid = _centroid(coords)
        self.normal = _plane_normal(coords)
        self.tag = tag


def _build_rings(atoms):
    rings = _find_rings(atoms)
    # Count rings per residue to disambiguate (e.g. TRP has two).
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
# Chemical feature classification (topology + residue names + formal charge)
# ===========================================================================

# Protein charged groups by residue/atom name.
_CATION_RES_ATOMS = {  # positively-charged centres
    "LYS": ["NZ"],
    "ARG": ["NH1", "NH2", "NE"],  # guanidinium -> averaged centre
    "HIS": ["ND1", "NE2"],  # protonated His (ambiguous; included)
    "HIP": ["ND1", "NE2"],
    "HSP": ["ND1", "NE2"],
}
_ANION_RES_ATOMS = {  # negatively-charged centres
    "ASP": ["OD1", "OD2"],
    "GLU": ["OE1", "OE2"],
}
_HALOGENS = {"Cl", "Br", "I"}  # F excluded: weak halogen-bond donor
_HB_ACCEPTOR_ELEMS = {"N", "O", "S", "F"}
_HB_DONOR_ELEMS = {"N", "O"}
# Coordination metals commonly seen in binding sites.
_METALS = {"Na", "K", "Mg", "Ca", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Cd", "Hg"}
# Water residue names (for water-mediated H-bonds).
_WATER_RESN = {"HOH", "WAT", "H2O", "SOL", "TIP", "TIP3", "TIP4", "SPC", "DOD"}

_CHEM_NON_ACCEPTOR_SYBYL = {"n.am", "n.pl3", "n.4"}
_CHEM_NON_DONOR_SYBYL = {"o.2", "o.co2"}
_PROTEIN_NON_ACCEPTOR = {
    ("ARG", "NE"),
    ("ARG", "NH1"),
    ("ARG", "NH2"),
    ("ASN", "ND2"),
    ("GLN", "NE2"),
    ("HIS", "ND1"),
    ("HIS", "NE2"),
    ("HIP", "ND1"),
    ("HIP", "NE2"),
    ("HID", "ND1"),
    ("HIE", "NE2"),
    ("HSP", "ND1"),
    ("HSP", "NE2"),
    ("LYS", "NZ"),
    ("TRP", "NE1"),
}
_PROTEIN_NON_DONOR_OXYGEN = {
    ("ASN", "OD1"),
    ("ASP", "OD1"),
    ("ASP", "OD2"),
    ("GLN", "OE1"),
    ("GLU", "OE1"),
    ("GLU", "OE2"),
}
_PROTEIN_HYDROXYL_DONORS = {
    ("SER", "OG"),
    ("THR", "OG1"),
    ("TYR", "OH"),
}
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
_BOND_ORDER_VALUES = {
    "1": 1.0,
    "2": 2.0,
    "3": 3.0,
    "ar": 1.5,
    "am": 1.0,
}


def _h_neighbors(atom):
    return [n for n in atom.neighbors if n.elem == "H"]


def _sybyl(atom):
    return str(getattr(atom, "sybyl_type", "") or "").strip().lower()


def _heavy_neighbors(atom):
    return [neighbor for neighbor in atom.neighbors if neighbor.elem != "H"]


def _ring_has_aromatic_evidence(ring):
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


def _chemistry_aware_alkyl_carbon(atom):
    residue = atom.resn.upper()
    if residue in _PROTEIN_RESIDUES:
        return atom.name.upper() in _PROTEIN_ALKYL_ATOMS.get(residue, set())
    heavy = _heavy_neighbors(atom)
    return bool(heavy) and all(neighbor.elem == "C" for neighbor in heavy)


def _strict_group_is_cationic(resname, atoms):
    if resname != "HIS":
        return True
    if any(atom.fcharge > 0 for atom in atoms):
        return True
    return len(atoms) >= 2 and all(_h_neighbors(atom) for atom in atoms)


def _heavy_bond_order_sum(atom):
    total = 0.0
    for neighbor in _heavy_neighbors(atom):
        raw_order = str(atom.bond_orders.get(neighbor.idx, "1")).lower()
        total += _BOND_ORDER_VALUES.get(raw_order, 1.0)
    return total


def _chemistry_aware_acceptor(atom):
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
        available_valence = len(heavy) <= 1 and _heavy_bond_order_sum(atom) <= 1.0
        return (sybyl == "o.3" and available_valence) or (
            not sybyl and available_valence and residue_atom in _PROTEIN_HYDROXYL_DONORS
        )
    if sybyl in {"n.1", "n.2", "n.ar", "n.4"}:
        return False
    if atom.resn.upper() == "PRO" and atom.name.upper() == "N":
        return False
    return len(heavy) < 3 and _heavy_bond_order_sum(atom) < 3.0


def classify(atoms, rings, has_h, chemistry_profile="plip"):
    """Return a dict of feature lists for one molecular side."""
    ring_atom_ids = set(a.idx for r in rings for a in r.atoms)

    donors = []  # (atom, [H atoms])  N/O-H donors
    carbon_donors = []  # (atom, [H atoms])  C-H donors
    sigma_donors = []  # (atom, [H atoms]) explicit C-H donors for pi-sigma
    acceptors = []  # atoms
    cations = []  # (point, label)
    anions = []  # (point, label)
    halogens = []  # (atom, bonded_C)  X-C
    alkyl_carbons = []  # sp3 aliphatic carbons
    metals = []  # metal ion atoms
    sulfurs = []  # S atoms (for pi-sulfur)
    chemistry_aware = str(chemistry_profile).strip().lower() == "dsv"
    side_has_explicit_hydrogens = any(atom.elem == "H" for atom in atoms)

    # --- charged centres from formal charge (ligands/ions) ---
    # Each charged centre is stored as (point, label, residue_selection).
    for a in atoms:
        if a.fcharge > 0:
            cations.append((a.coord, a.label(), a.res_sele()))
        elif a.fcharge < 0:
            anions.append((a.coord, a.label(), a.res_sele()))

    # --- protein charged groups (grouped centres) ---
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
        cations.append((pt, lbl, grp[0].res_sele()))
    for (res, _resn), grp in grouped_anion.items():
        pt = _centroid([x.coord for x in grp])
        anions.append((pt, "%s_carboxyl" % res, grp[0].res_sele()))

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
                cations.append((atom.coord, atom.label(), atom.res_sele()))

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
            anions.append(
                (point, "%s_carboxylate" % group[0].res_tag(), group[0].res_sele())
            )

    # --- H-bond donors/acceptors, halogens, alkyl carbons ---
    for a in atoms:
        if chemistry_aware:
            if _chemistry_aware_acceptor(a):
                acceptors.append(a)
            if _chemistry_aware_donor(
                a,
                allow_inferred_hydrogen=not side_has_explicit_hydrogens,
            ):
                donors.append((a, _h_neighbors(a)))
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
                    donors.append((a, []))  # heavy-atom-only mode
        if a.elem == "C":
            if has_h:
                hs = _h_neighbors(a)
                heavy_neighbors = _heavy_neighbors(a)
                polarized = any(
                    neighbor.elem in {"N", "O", "S", "F", "Cl", "Br", "I"}
                    for neighbor in heavy_neighbors
                )
                if hs:
                    if not chemistry_aware or polarized:
                        carbon_donors.append((a, hs))
                    if _sybyl(a) in {"", "c.3"}:
                        sigma_donors.append((a, hs))
            # aliphatic carbon: not aromatic-ring member, bonded only to C/H
            if a.idx not in ring_atom_ids:
                if chemistry_aware:
                    if _chemistry_aware_alkyl_carbon(a):
                        alkyl_carbons.append(a)
                else:
                    heavy = _heavy_neighbors(a)
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

    return {
        "donors": donors,
        "carbon_donors": carbon_donors,
        "sigma_donors": sigma_donors,
        "acceptors": acceptors,
        "cations": cations,
        "anions": anions,
        "halogens": halogens,
        "alkyl": alkyl_carbons,
        "metals": metals,
        "sulfurs": sulfurs,
        "rings": rings,
    }


# ===========================================================================
# Interaction detectors
# ===========================================================================
# Each detector yields interaction dicts:
#   {type, subtype, a_label, b_label, a_point, b_point}
# a_point/b_point are numpy coords used to draw the dashed line.


def _hbond_pairs(feat_a, feat_b, itype, dist_cut, angle_cut, has_h):
    """Shared donor->acceptor logic for conventional and carbon H-bonds."""
    donor_key = "donors" if itype == "hbond" else "carbon_donors"
    out = []
    for donor, hs in feat_a[donor_key]:
        for acc in feat_b["acceptors"]:
            if donor.idx == acc.idx:
                continue
            d = _dist(donor.coord, acc.coord)
            if d > dist_cut:
                continue
            dsv_engine = _active_engine[0] == "ds"
            if dsv_engine and hs:
                prefix = "hbond" if itype == "hbond" else "carbon_hbond"
                acceptor_bases = [
                    neighbor for neighbor in acc.neighbors if neighbor.elem != "H"
                ]
                if not acceptor_bases:
                    continue
                for hydrogen in hs:
                    hydrogen_distance = _dist(hydrogen.coord, acc.coord)
                    if hydrogen_distance > CUTOFFS["%s_h_a_dist" % prefix]:
                        continue
                    donor_angle = _angle_at(hydrogen.coord, donor.coord, acc.coord)
                    if donor_angle < angle_cut:
                        continue
                    acceptor_angle = max(
                        _angle_at(acc.coord, hydrogen.coord, base.coord)
                        for base in acceptor_bases
                    )
                    if acceptor_angle < CUTOFFS["%s_acceptor_angle" % prefix]:
                        continue
                    out.append(
                        {
                            "type": itype,
                            "subtype": "",
                            "a_label": donor.label(),
                            "b_label": acc.label(),
                            "a_point": donor.coord,
                            "b_point": acc.coord,
                            "a_sele": donor.res_sele(),
                            "b_sele": acc.res_sele(),
                            "chemistry_basis": "explicit_hydrogen",
                            "confidence": "high",
                            "hydrogen": hydrogen.label(),
                            "hydrogen_acceptor_distance_A": hydrogen_distance,
                            "donor_hydrogen_acceptor_angle_deg": donor_angle,
                            "hydrogen_acceptor_base_angle_deg": acceptor_angle,
                        }
                    )
                continue
            if dsv_engine and not hs:
                prefix = "hbond" if itype == "hbond" else "carbon_hbond"
                if d > CUTOFFS["%s_inferred_dist" % prefix]:
                    continue
            if not dsv_engine and has_h and hs:
                best = max(_angle_at(h.coord, donor.coord, acc.coord) for h in hs)
                if best < angle_cut:
                    continue
            chemistry_metadata = {}
            if dsv_engine:
                chemistry_metadata = {
                    "chemistry_basis": "inferred_hydrogen",
                    "confidence": "medium",
                }
            out.append(
                {
                    "type": itype,
                    "subtype": "",
                    "a_label": donor.label(),
                    "b_label": acc.label(),
                    "a_point": donor.coord,
                    "b_point": acc.coord,
                    "a_sele": donor.res_sele(),
                    "b_sele": acc.res_sele(),
                    **chemistry_metadata,
                }
            )
    return out


def detect_hbond(fa, fb, has_h):
    c = CUTOFFS
    res = _hbond_pairs(fa, fb, "hbond", c["hbond_dist"], c["hbond_angle"], has_h)
    res += _hbond_pairs(fb, fa, "hbond", c["hbond_dist"], c["hbond_angle"], has_h)
    return res


def detect_carbon_hbond(fa, fb, has_h):
    c = CUTOFFS
    res = _hbond_pairs(
        fa, fb, "carbon_hbond", c["carbon_hbond_dist"], c["carbon_hbond_angle"], has_h
    )
    res += _hbond_pairs(
        fb, fa, "carbon_hbond", c["carbon_hbond_dist"], c["carbon_hbond_angle"], has_h
    )
    return res


def detect_saltbridge(fa, fb):
    cut = CUTOFFS["saltbridge_dist"]
    out = []
    for cats, anis in ((fa["cations"], fb["anions"]), (fb["cations"], fa["anions"])):
        for cpt, clbl, csele in cats:
            for apt, albl, asele in anis:
                if _dist(cpt, apt) <= cut:
                    out.append(
                        {
                            "type": "saltbridge",
                            "subtype": "",
                            "a_label": clbl,
                            "b_label": albl,
                            "a_point": cpt,
                            "b_point": apt,
                            "a_sele": csele,
                            "b_sele": asele,
                        }
                    )
    return out


def detect_pipi(fa, fb):
    c = CUTOFFS
    out = []
    for r1 in fa["rings"]:
        for r2 in fb["rings"]:
            if _dist(r1.centroid, r2.centroid) > c["pipi_dist"]:
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
                {
                    "type": "pipi",
                    "subtype": subtype,
                    "a_label": r1.tag,
                    "b_label": r2.tag,
                    "a_point": r1.centroid,
                    "b_point": r2.centroid,
                    "a_sele": r1.atoms[0].res_sele(),
                    "b_sele": r2.atoms[0].res_sele(),
                }
            )
    return out


def detect_pication(fa, fb):
    c = CUTOFFS
    out = []
    for rings, cats in ((fa["rings"], fb["cations"]), (fb["rings"], fa["cations"])):
        for r in rings:
            for cpt, clbl, csele in cats:
                if _dist(r.centroid, cpt) > c["pication_dist"]:
                    continue
                if _proj_offset(cpt, r.centroid, r.normal) > c["pication_offset"]:
                    continue
                out.append(
                    {
                        "type": "pication",
                        "subtype": "",
                        "a_label": r.tag,
                        "b_label": clbl,
                        "a_point": r.centroid,
                        "b_point": cpt,
                        "a_sele": r.atoms[0].res_sele(),
                        "b_sele": csele,
                    }
                )
    return out


def detect_pialkyl(fa, fb):
    cut = CUTOFFS["pialkyl_dist"]
    out = []
    best_by_group = {}
    dsv_engine = _active_engine[0] == "ds"
    for rings, alks in ((fa["rings"], fb["alkyl"]), (fb["rings"], fa["alkyl"])):
        for r in rings:
            for a in alks:
                distance = _dist(r.centroid, a.coord)
                if distance > cut:
                    continue
                if dsv_engine and any(
                    _pi_sigma_geometry(r, a, h) is not None for h in _h_neighbors(a)
                ):
                    continue
                record = {
                    "type": "pialkyl",
                    "subtype": "",
                    "a_label": r.tag,
                    "b_label": a.label(),
                    "a_point": r.centroid,
                    "b_point": a.coord,
                    "a_sele": r.atoms[0].res_sele(),
                    "b_sele": a.res_sele(),
                }
                if not dsv_engine:
                    out.append(record)
                    continue
                key = (r.tag, a.res_tag())
                previous = best_by_group.get(key)
                if previous is None or distance < previous[0]:
                    best_by_group[key] = (distance, record)
    if dsv_engine:
        out.extend(value[1] for value in best_by_group.values())
    return out


def _pi_sigma_geometry(ring, donor, hydrogen):
    """Return calibrated C-H/pi geometry or None when it fails."""
    donor_distance = _dist(ring.centroid, donor.coord)
    hydrogen_distance = _dist(ring.centroid, hydrogen.coord)
    theta = _axis_angle(hydrogen.coord, ring.centroid, ring.normal)
    donor_angle = _angle_at(hydrogen.coord, donor.coord, ring.centroid)
    c = CUTOFFS
    if (
        donor_distance > c["pi_sigma_carbon_dist"]
        or hydrogen_distance > c["pi_sigma_h_centroid_dist"]
        or theta > c["pi_sigma_axis_angle"]
        or donor_angle < c["pi_sigma_dha_angle"]
    ):
        return None
    return donor_distance, hydrogen_distance, theta, donor_angle


def detect_pi_sigma(fa, fb):
    """Detect an axial explicit C-H sigma bond pointing at an aromatic face."""
    if _active_engine[0] != "ds":
        return []
    best_by_pair = {}
    for rings, donors in ((fa["rings"], fb["sigma_donors"]),
                          (fb["rings"], fa["sigma_donors"])):
        for ring in rings:
            for donor, hydrogens in donors:
                for hydrogen in hydrogens:
                    geometry = _pi_sigma_geometry(ring, donor, hydrogen)
                    if geometry is None:
                        continue
                    donor_distance, h_distance, theta, dha = geometry
                    record = {
                        "type": "pi_sigma", "subtype": "C-H/pi",
                        "a_label": ring.tag, "b_label": donor.label(),
                        "a_point": ring.centroid, "b_point": donor.coord,
                        "a_sele": ring.atoms[0].res_sele(),
                        "b_sele": donor.res_sele(), "hydrogen": hydrogen.label(),
                        "hydrogen_centroid_distance_A": h_distance,
                        "donor_hydrogen_centroid_angle_deg": dha,
                        "theta_deg": theta,
                        "donor_centroid_distance_A": donor_distance,
                    }
                    key = (ring.tag, donor.idx)
                    if key not in best_by_pair or h_distance < best_by_pair[key][0]:
                        best_by_pair[key] = (h_distance, record)
    return [value[1] for value in best_by_pair.values()]


def detect_pi_donor_hbond(fa, fb):
    """Detect an explicit N/O-H donor directed towards an aromatic face."""
    if _active_engine[0] != "ds":
        return []
    c = CUTOFFS
    best_by_pair = {}
    for rings, donors in ((fa["rings"], fb["donors"]), (fb["rings"], fa["donors"])):
        for ring in rings:
            for donor, hydrogens in donors:
                for hydrogen in hydrogens:
                    donor_distance = _dist(ring.centroid, donor.coord)
                    h_distance = _dist(ring.centroid, hydrogen.coord)
                    theta = _axis_angle(hydrogen.coord, ring.centroid, ring.normal)
                    dha = _angle_at(hydrogen.coord, donor.coord, ring.centroid)
                    if (donor_distance > c["pi_donor_dist"] or
                            h_distance > c["pi_donor_h_centroid_dist"] or
                            theta > c["pi_donor_axis_angle"] or
                            dha < c["pi_donor_dha_angle"]):
                        continue
                    record = {
                        "type": "pi_donor_hbond", "subtype": "X-H/pi",
                        "a_label": ring.tag, "b_label": donor.label(),
                        "a_point": ring.centroid, "b_point": donor.coord,
                        "a_sele": ring.atoms[0].res_sele(),
                        "b_sele": donor.res_sele(), "hydrogen": hydrogen.label(),
                        "hydrogen_centroid_distance_A": h_distance,
                        "donor_hydrogen_centroid_angle_deg": dha,
                        "theta_deg": theta,
                        "donor_centroid_distance_A": donor_distance,
                    }
                    key = (ring.tag, donor.idx)
                    if key not in best_by_pair or h_distance < best_by_pair[key][0]:
                        best_by_pair[key] = (h_distance, record)
    return [value[1] for value in best_by_pair.values()]


def detect_alkyl(fa, fb):
    cut = CUTOFFS["alkyl_dist"]
    out = []
    for a in fa["alkyl"]:
        for b in fb["alkyl"]:
            if _dist(a.coord, b.coord) <= cut:
                out.append(
                    {
                        "type": "alkyl",
                        "subtype": "",
                        "a_label": a.label(),
                        "b_label": b.label(),
                        "a_point": a.coord,
                        "b_point": b.coord,
                        "a_sele": a.res_sele(),
                        "b_sele": b.res_sele(),
                    }
                )
    return out


def detect_halogen(fa, fb):
    c = CUTOFFS
    out = []
    for hals, accs in (
        (fa["halogens"], fb["acceptors"]),
        (fb["halogens"], fa["acceptors"]),
    ):
        for x, cbonded in hals:
            for acc in accs:
                if _dist(x.coord, acc.coord) > c["halogen_dist"]:
                    continue
                # C-X...A angle (halogen bonds are near-linear at X).
                if _angle_at(x.coord, cbonded.coord, acc.coord) < c["halogen_angle"]:
                    continue
                out.append(
                    {
                        "type": "halogen",
                        "subtype": "",
                        "a_label": x.label(),
                        "b_label": acc.label(),
                        "a_point": x.coord,
                        "b_point": acc.coord,
                        "a_sele": x.res_sele(),
                        "b_sele": acc.res_sele(),
                    }
                )
    return out


def detect_metal(fa, fb):
    cut = CUTOFFS["metal_dist"]
    out = []
    for metals, accs in (
        (fa["metals"], fb["acceptors"]),
        (fb["metals"], fa["acceptors"]),
    ):
        for m in metals:
            for acc in accs:
                if _dist(m.coord, acc.coord) <= cut:
                    out.append(
                        {
                            "type": "metal",
                            "subtype": "",
                            "a_label": m.label(),
                            "b_label": acc.label(),
                            "a_point": m.coord,
                            "b_point": acc.coord,
                            "a_sele": m.res_sele(),
                            "b_sele": acc.res_sele(),
                        }
                    )
    return out


def detect_pi_sulfur(fa, fb):
    cut = CUTOFFS["pi_sulfur_dist"]
    out = []
    dsv_engine = _active_engine[0] == "ds"
    for rings, sulfs in ((fa["rings"], fb["sulfurs"]), (fb["rings"], fa["sulfurs"])):
        for r in rings:
            if dsv_engine and not _ring_has_aromatic_evidence(r):
                continue
            for s in sulfs:
                if _dist(r.centroid, s.coord) <= cut:
                    out.append(
                        {
                            "type": "pi_sulfur",
                            "subtype": "",
                            "a_label": r.tag,
                            "b_label": s.label(),
                            "a_point": r.centroid,
                            "b_point": s.coord,
                            "a_sele": r.atoms[0].res_sele(),
                            "b_sele": s.res_sele(),
                        }
                    )
    return out


def detect_pi_anion(fa, fb):
    c = CUTOFFS
    out = []
    for rings, anis in ((fa["rings"], fb["anions"]), (fb["rings"], fa["anions"])):
        for r in rings:
            for apt, albl, asele in anis:
                if _dist(r.centroid, apt) > c["pi_anion_dist"]:
                    continue
                if _proj_offset(apt, r.centroid, r.normal) > c["pi_anion_offset"]:
                    continue
                out.append(
                    {
                        "type": "pi_anion",
                        "subtype": "",
                        "a_label": r.tag,
                        "b_label": albl,
                        "a_point": r.centroid,
                        "b_point": apt,
                        "a_sele": r.atoms[0].res_sele(),
                        "b_sele": asele,
                    }
                )
    return out


def detect_pi_lone_pair(fa, fb):
    """Detect an acceptor lone-pair positioned above an aromatic face."""
    if _active_engine[0] != "ds":
        return []
    c = CUTOFFS
    out = []
    for acceptors, rings in ((fa["acceptors"], fb["rings"]),
                             (fb["acceptors"], fa["rings"])):
        for acceptor in acceptors:
            for ring in rings:
                if not _ring_has_aromatic_evidence(ring):
                    continue
                distance = _dist(acceptor.coord, ring.centroid)
                if distance > c["pi_lone_pair_dist"]:
                    continue
                direction = _v(acceptor.coord) - _v(ring.centroid)
                norm = np.linalg.norm(direction)
                if norm < 1e-6:
                    continue
                theta = _axis_angle(acceptor.coord, ring.centroid, ring.normal)
                if theta > c["pi_lone_pair_angle"]:
                    continue
                out.append(
                    {
                        "type": "pi_lone_pair", "subtype": "lone-pair/pi",
                        "a_label": acceptor.label(), "b_label": ring.tag,
                        "a_point": acceptor.coord, "b_point": ring.centroid,
                        "a_sele": acceptor.res_sele(),
                        "b_sele": ring.atoms[0].res_sele(), "theta_deg": theta,
                    }
                )
    return out


def detect_water_bridge(fa, fb, waters):
    """Water-mediated H-bond: an atom in fa and an atom in fb both H-bond to the
    same bridging water. Emits two dashed legs (partner--water) per bridge.

    `waters` is a list of water-oxygen Atom objects. A partner is any donor or
    acceptor heavy atom (N/O) already collected on each side.
    """
    c = CUTOFFS

    def _partners(feat):
        seen, out = set(), []
        for donor, _hs in feat["donors"]:
            if donor.idx not in seen:
                seen.add(donor.idx)
                out.append(donor)
        for acc in feat["acceptors"]:
            if acc.idx not in seen:
                seen.add(acc.idx)
                out.append(acc)
        return out

    pa, pb = _partners(fa), _partners(fb)
    lo, hi = c["water_bridge_min"], c["water_bridge_max"]
    amin, amax = c["water_bridge_angle_min"], c["water_bridge_angle_max"]
    out = []
    for w in waters:
        near_a = [p for p in pa if lo <= _dist(w.coord, p.coord) <= hi]
        near_b = [p for p in pb if lo <= _dist(w.coord, p.coord) <= hi]
        for pai in near_a:
            for pbi in near_b:
                ang = _angle_at(w.coord, pai.coord, pbi.coord)
                if not (amin <= ang <= amax):
                    continue
                wlbl = w.label()
                for partner in (pai, pbi):
                    out.append(
                        {
                            "type": "water_bridge",
                            "subtype": "",
                            "a_label": partner.label(),
                            "b_label": wlbl,
                            "a_point": partner.coord,
                            "b_point": w.coord,
                            "a_sele": partner.res_sele(),
                            "b_sele": w.res_sele(),
                        }
                    )
    return out


_DETECTORS = {
    "hbond": lambda fa, fb, h: detect_hbond(fa, fb, h),
    "carbon_hbond": lambda fa, fb, h: detect_carbon_hbond(fa, fb, h),
    "saltbridge": lambda fa, fb, h: detect_saltbridge(fa, fb),
    "pipi": lambda fa, fb, h: detect_pipi(fa, fb),
    "pication": lambda fa, fb, h: detect_pication(fa, fb),
    "pialkyl": lambda fa, fb, h: detect_pialkyl(fa, fb),
    "alkyl": lambda fa, fb, h: detect_alkyl(fa, fb),
    "halogen": lambda fa, fb, h: detect_halogen(fa, fb),
    "metal": lambda fa, fb, h: detect_metal(fa, fb),
    "pi_sulfur": lambda fa, fb, h: detect_pi_sulfur(fa, fb),
    "pi_anion": lambda fa, fb, h: detect_pi_anion(fa, fb),
    "pi_sigma": lambda fa, fb, h: detect_pi_sigma(fa, fb),
    "pi_donor_hbond": lambda fa, fb, h: detect_pi_donor_hbond(fa, fb),
    "pi_lone_pair": lambda fa, fb, h: detect_pi_lone_pair(fa, fb),
    # water_bridge handled separately (needs the water list) in _compute.
}


# ---------------------------------------------------------------------------
# DockLens parity adapter
# ---------------------------------------------------------------------------
# The historical detector implementations above are retained for source
# compatibility. Public detection is rebound here to the bundled, hash-identical
# DockLens core, so drawing, counting, occupancy and CSV all consume the same
# scientific records.


def _chemistry_profile_for_engine():
    return "dsv" if _active_engine[0] == "ds" else "plip"


def _endpoint_atom(obj):
    return obj.atoms[0] if hasattr(obj, "atoms") else obj


def _endpoint_selection(obj):
    return _endpoint_atom(obj).res_sele()


def _adapt_docklens_record(interaction):
    record = dict(interaction)
    record["a_sele"] = _endpoint_selection(record["a_obj"])
    record["b_sele"] = _endpoint_selection(record["b_obj"])

    hydrogen = record.get("hydrogen_obj")
    if hydrogen is not None:
        record["hydrogen"] = hydrogen.label()
        if record["type"] in {
            "hbond",
            "carbon_hbond",
            "pi_sigma",
            "pi_donor_hbond",
        }:
            donor_prefix = None
            for prefix in ("a", "b"):
                if "donor" in record.get("%s_role" % prefix, ""):
                    donor_prefix = prefix
                    break
            if donor_prefix is not None:
                # Discovery Studio draws explicit-H contacts from H to the
                # acceptor/ring. Preserve the heavy donor in a_obj/b_obj for
                # chemistry and CSV normalization, but use H for the figure.
                record["%s_label" % donor_prefix] = hydrogen.label()
                record["%s_point" % donor_prefix] = hydrogen.coord
    hydrogen_distance = record.get("hydrogen_acceptor_distance")
    donor_angle = record.get("donor_hydrogen_acceptor_angle")
    acceptor_angle = record.get("hydrogen_acceptor_base_angle")
    if record["type"] in {"pi_sigma", "pi_donor_hbond"}:
        record["hydrogen_centroid_distance_A"] = hydrogen_distance
        record["donor_hydrogen_centroid_angle_deg"] = donor_angle
    else:
        record["hydrogen_acceptor_distance_A"] = hydrogen_distance
        record["donor_hydrogen_acceptor_angle_deg"] = donor_angle
        record["hydrogen_acceptor_base_angle_deg"] = acceptor_angle
    if "theta" in record:
        record["theta_deg"] = record["theta"]
    return record


def _run_docklens_detector(name, feat_a, feat_b, has_h=False):
    cutoff_token = _docklens_core._ACTIVE_CUTOFFS.set(dict(CUTOFFS))
    chemistry_token = _docklens_core._ACTIVE_CHEMISTRY_PROFILE.set(
        _chemistry_profile_for_engine()
    )
    try:
        detector = getattr(_docklens_core, name)
        if name in {"detect_hbond", "detect_carbon_hbond"}:
            records = detector(feat_a, feat_b, has_h)
        else:
            records = detector(feat_a, feat_b)
        return [_adapt_docklens_record(record) for record in records]
    finally:
        _docklens_core._ACTIVE_CHEMISTRY_PROFILE.reset(chemistry_token)
        _docklens_core._ACTIVE_CUTOFFS.reset(cutoff_token)


def classify(atoms, rings, has_h, chemistry_profile=None):  # noqa: F811
    profile = chemistry_profile or _chemistry_profile_for_engine()
    return _docklens_core.classify(
        atoms,
        rings,
        has_h,
        chemistry_profile=profile,
    )


Ring = _docklens_core.Ring  # noqa: F811


def detect_hbond(fa, fb, has_h):  # noqa: F811
    return _run_docklens_detector("detect_hbond", fa, fb, has_h)


def detect_carbon_hbond(fa, fb, has_h):  # noqa: F811
    return _run_docklens_detector("detect_carbon_hbond", fa, fb, has_h)


def detect_saltbridge(fa, fb):  # noqa: F811
    return _run_docklens_detector("detect_saltbridge", fa, fb)


def detect_pipi(fa, fb):  # noqa: F811
    return _run_docklens_detector("detect_pipi", fa, fb)


def detect_pication(fa, fb):  # noqa: F811
    return _run_docklens_detector("detect_pication", fa, fb)


def detect_pialkyl(fa, fb):  # noqa: F811
    return _run_docklens_detector("detect_pialkyl", fa, fb)


def detect_pi_sigma(fa, fb):  # noqa: F811
    return _run_docklens_detector("detect_pi_sigma", fa, fb)


def detect_alkyl(fa, fb):  # noqa: F811
    return _run_docklens_detector("detect_alkyl", fa, fb)


def detect_halogen(fa, fb):  # noqa: F811
    return _run_docklens_detector("detect_halogen", fa, fb)


def detect_metal(fa, fb):  # noqa: F811
    return _run_docklens_detector("detect_metal", fa, fb)


def detect_pi_sulfur(fa, fb):  # noqa: F811
    return _run_docklens_detector("detect_pi_sulfur", fa, fb)


def detect_pi_anion(fa, fb):  # noqa: F811
    return _run_docklens_detector("detect_pi_anion", fa, fb)


def detect_pi_donor_hbond(fa, fb):  # noqa: F811
    return _run_docklens_detector("detect_pi_donor_hbond", fa, fb)


def detect_pi_lone_pair(fa, fb):  # noqa: F811
    return _run_docklens_detector("detect_pi_lone_pair", fa, fb)


_DETECTORS = {
    "hbond": lambda fa, fb, h: detect_hbond(fa, fb, h),
    "carbon_hbond": lambda fa, fb, h: detect_carbon_hbond(fa, fb, h),
    "saltbridge": lambda fa, fb, h: detect_saltbridge(fa, fb),
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
}


def _matches_ds_like(interaction):
    detail = types.SimpleNamespace(
        interaction_type=interaction["type"],
        distance_A=interaction.get("dist"),
    )
    return _docklens_analysis_profiles.detail_matches_profile(detail, "ds_like")


# ===========================================================================
# Drawing
# ===========================================================================

_HELPER = "_ii_pts"  # hidden object holding centroid/endpoint pseudoatoms
_PSEUDO_COUNTER = [0]
_drawn_names = set()
# name -> (base_dash_length, base_dash_gap); lets the appearance controls scale
# the per-type dash pattern without losing the sandwich/T-shaped/dotted encoding.
_dash_base = {}


def _sanitize(name):
    return "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in name)


def _pseudo_at(point):
    """Create a hidden pseudoatom at `point`; return a selection string for it."""
    _PSEUDO_COUNTER[0] += 1
    n = _PSEUDO_COUNTER[0]
    cmd.pseudoatom(
        _HELPER,
        resi=str(n),
        name="PS",
        pos=[float(point[0]), float(point[1]), float(point[2])],
    )
    return "%s and resi %d" % (_HELPER, n)


def _draw(interaction, group_name, keep_label, _water_leg=False):
    if (
        interaction["type"] == "water_bridge"
        and interaction.get("water_obj") is not None
        and not _water_leg
    ):
        water = interaction["water_obj"]
        first_leg = dict(interaction)
        first_leg.update(
            {
                "b_label": water.label(),
                "b_point": water.coord,
                "b_sele": water.res_sele(),
            }
        )
        second_leg = dict(interaction)
        second_leg.update(
            {
                "a_label": water.label(),
                "a_point": water.coord,
                "a_sele": water.res_sele(),
            }
        )
        return (
            _draw(first_leg, group_name, keep_label, _water_leg=True),
            _draw(second_leg, group_name, keep_label, _water_leg=True),
        )

    itype = interaction["type"]
    subtype = interaction["subtype"]
    name = _sanitize(
        "%s--%s_%s" % (interaction["a_label"], interaction["b_label"], itype)
    )
    # Disambiguate identical names (rare duplicate labels).
    base, k = name, 1
    while name in _drawn_names:
        k += 1
        name = "%s_%d" % (base, k)
    _drawn_names.add(name)

    sa = _pseudo_at(interaction["a_point"])
    sb = _pseudo_at(interaction["b_point"])
    cmd.distance(name, sa, sb)

    cmd.set("dash_color", _color_name(itype), name)
    if itype == "pipi" and subtype == "tshaped":
        dl, dg, dr = _PIPI_TSHAPED_DASH
    elif itype == "pipi":
        dl, dg, dr = _PIPI_SANDWICH_DASH
    else:
        dl, dg, dr = _DASH_STYLE[itype]
    cmd.set("dash_length", dl, name)
    cmd.set("dash_gap", dg, name)
    cmd.set("dash_radius", dr, name)
    _dash_base[name] = (dl, dg)
    if not keep_label:
        cmd.hide("labels", name)
    cmd.group(group_name, name)
    return name


# ===========================================================================
# Public commands
# ===========================================================================


def _parse_types(types):
    if types in (None, "", "all", "All", "ALL"):
        return list(VALID_TYPES)
    if isinstance(types, (list, tuple)):
        req = list(types)
    else:
        req = str(types).replace(",", " ").split()
    bad = [t for t in req if t not in VALID_TYPES]
    if bad:
        raise cmd.QuietException(
            "Unknown interaction type(s): %s. Valid: %s"
            % (", ".join(bad), ", ".join(VALID_TYPES))
        )
    return req


def _disable_native_hbonds():
    """Hide PyMOL's own polar-contact/H-bond dash objects (never deletes)."""
    hidden = []
    for obj in cmd.get_names("objects"):
        low = obj.lower()
        if (
            "polar_contact" in low
            or "_hbond" in low
            or "hbonds" in low
            or low.endswith("_hb")
        ):
            cmd.disable(obj)
            hidden.append(obj)
    return hidden


def _show_residue_sticks(sel1, sel2, res_seles, group_name):
    """Show every interacting residue as sticks in a named selection.

    Builds `<group_name>_residues` = (sel1 or sel2) intersected with the union
    of the residue selections that took part in an interaction, then displays it
    as sticks (carbons coloured for contrast). Returns the selection name.
    """
    ressel = "%s_residues" % group_name
    combined = "(%s) or (%s)" % (sel1, sel2)
    expr = "(%s) and (%s)" % (combined, " or ".join(sorted(res_seles)))
    cmd.select(ressel, expr)
    cmd.show("sticks", ressel)
    try:
        cmd.util.cnc(ressel)  # colour non-carbon atoms by element
    except Exception:
        pass
    cmd.deselect()  # drop the pink selection markers
    return ressel


def _resolve_selection(sel):
    """Resolve 'auto' to a best-guess ligand selection; pass others through."""
    if str(sel).strip().lower() != "auto":
        return sel
    if cmd.count_atoms("organic") > 0:
        return "organic"
    waters = "+".join(sorted(_WATER_RESN))
    return "(not polymer) and (not resn %s) and (not elem %s)" % (
        waters,
        "+".join(sorted(_METALS)),
    )


def _resolve_receptor_selection(sel):
    """Make the DS default match DockLens' receptor scope, including metals."""
    value = str(sel).strip()
    if _active_engine[0] == "ds" and value.lower() == "polymer":
        return "(polymer or metals)"
    return sel


def _validate_group_name(group_name):
    """Protect PyMOL sessions from broad or malformed delete targets."""
    value = str(group_name or "").strip()
    if (
        not _GROUP_NAME_PATTERN.fullmatch(value)
        or value.lower() in _RESERVED_GROUP_NAMES
    ):
        raise ValueError(
            "group_name must be a simple, non-reserved PyMOL object name"
        )
    return value


def _load_waters(sel1, sel2, state, index_offset=0):
    """Water-oxygen Atom objects near either selection (for water bridges)."""
    resn = "+".join(sorted(_WATER_RESN))
    near = "(resn %s) and elem O and (byres (all within 5 of ((%s) or (%s))))" % (
        resn,
        sel1,
        sel2,
    )
    if cmd.count_atoms(near) == 0:
        return []
    waters, _has_h = _load_atoms(near, state, index_offset=index_offset)
    return waters


def _compute_interactions(sel1, sel2, req_types, state):
    """Detect (but do NOT draw) all requested interactions for one state.

    Returns (interactions, has_h). Each interaction dict carries a computed
    'dist' (endpoint separation, Angstrom). Shared by detect_interactions,
    interactions_occupancy and interactions_export_csv.
    """
    _last_compute_cost[0] = 0
    atoms1, has_h1 = _load_atoms(sel1, state, index_offset=0)
    atoms2, has_h2 = _load_atoms(sel2, state, index_offset=len(atoms1))
    if not atoms1 or not atoms2:
        return [], None
    if len(atoms1) * len(atoms2) > MAX_ATOM_PAIRS:
        raise ValueError(
            "Selections are too large for an all-pairs analysis "
            "(%d x %d atoms; limit %d pairs). Narrow sel1/sel2."
            % (len(atoms1), len(atoms2), MAX_ATOM_PAIRS)
        )
    atom_keys_1 = {
        (getattr(atom, "model", ""), getattr(atom, "pymol_index", None))
        for atom in atoms1
        if getattr(atom, "model", "")
        and getattr(atom, "pymol_index", None) is not None
    }
    atom_keys_2 = {
        (getattr(atom, "model", ""), getattr(atom, "pymol_index", None))
        for atom in atoms2
        if getattr(atom, "model", "")
        and getattr(atom, "pymol_index", None) is not None
    }
    if atom_keys_1.intersection(atom_keys_2):
        raise ValueError("sel1 and sel2 must be distinct, non-overlapping groups")
    has_h = has_h1 or has_h2
    chemistry_profile = "dsv" if _active_engine[0] == "ds" else "plip"
    for atom in atoms1:
        atom.side = "receptor"
    for atom in atoms2:
        atom.side = "ligand"
    waters = (
        _load_waters(
            sel1,
            sel2,
            state,
            index_offset=len(atoms1) + len(atoms2),
        )
        if "water_bridge" in req_types
        else []
    )
    for atom in waters:
        atom.side = "water"
    compute_cost = _estimate_compute_cost(
        len(atoms1),
        len(atoms2),
        len(waters),
        include_water="water_bridge" in req_types,
    )
    _last_compute_cost[0] = compute_cost
    if compute_cost > MAX_COMPUTE_COST_PER_FRAME:
        raise ValueError(
            "Selections and waters exceed the per-frame processing budget "
            "(estimated %d operations; limit %d)."
            % (compute_cost, MAX_COMPUTE_COST_PER_FRAME)
        )

    raw_interactions = _docklens_core.compute_interactions(
        atoms1,
        atoms2,
        waters=waters,
        types=req_types,
        cutoffs=dict(CUTOFFS),
        chemistry_profile=chemistry_profile,
    )
    inters = [_adapt_docklens_record(record) for record in raw_interactions]
    if chemistry_profile == "dsv":
        inters = [interaction for interaction in inters if _matches_ds_like(interaction)]
        diagnostics = []
        all_atoms = atoms1 + atoms2
        if not any(atom.sybyl_type for atom in all_atoms):
            diagnostics.append(
                "SYBYL atom types are unavailable; DockLens PDB fallback is active"
            )
        if not any(atom.bond_orders for atom in all_atoms):
            diagnostics.append(
                "bond orders are unavailable; aromatic/valence evidence may be limited"
            )
        _last_parity_diagnostics[:] = diagnostics
    else:
        _last_parity_diagnostics[:] = []
    return inters, has_h


def detect_interactions(
    sel1="polymer",
    sel2="organic",
    types="all",
    state=1,
    disable_native_hbond=1,
    group_name="interactions",
    label=0,
    show_residues=0,
    engine="",
):
    """Detect and draw non-covalent interactions between sel1 and sel2.

    See the module docstring for full parameter documentation and examples.
    Re-run with a different `state` to recompute a specific MD frame; the
    interaction group is rebuilt from scratch on every call.

    sel2='auto' picks the ligand automatically (organic, else non-polymer).
    show_residues 1 => also display interacting residues as sticks in a
        selection named '<group_name>_residues'.
    engine  '' (default) => keep the currently active engine; 'plip' or
        'ds' => switch engine first (see interactions_set_engine).
    """
    if engine:
        interactions_set_engine(engine)
    _register_colors()
    sel1 = _resolve_receptor_selection(_resolve_selection(sel1))
    sel2 = _resolve_selection(sel2)
    state = int(state)
    keep_label = int(label) != 0
    do_disable = int(disable_native_hbond) != 0
    do_residues = int(show_residues) != 0
    req_types = _parse_types(types)
    group_name = _validate_group_name(group_name)

    inters, has_h = _compute_interactions(sel1, sel2, req_types, state)
    if has_h is None:
        print("[interactions] empty selection (sel1 or sel2 has no atoms).")
        return
    if len(inters) > MAX_DRAWN_INTERACTIONS:
        raise ValueError(
            "Analysis found %d interactions; drawing is limited to %d. "
            "Narrow the selections or export CSV instead."
            % (len(inters), MAX_DRAWN_INTERACTIONS)
        )

    # Fresh slate for this group + helper (idempotent re-run / per-frame redraw).
    cmd.delete(group_name)
    cmd.delete(_HELPER)
    cmd.delete("%s_residues" % group_name)
    _PSEUDO_COUNTER[0] = 0
    _drawn_names.clear()
    _dash_base.clear()

    if do_disable:
        _disable_native_hbonds()

    counts = {}
    res_seles = set()
    for inter in inters:
        _draw(inter, group_name, keep_label)
        counts[inter["type"]] = counts.get(inter["type"], 0) + 1
        res_seles.add(inter["a_sele"])
        res_seles.add(inter["b_sele"])

    cmd.disable(_HELPER)  # keep endpoint pseudoatoms hidden

    if do_residues and res_seles:
        _show_residue_sticks(sel1, sel2, res_seles, group_name)

    total = sum(counts.values())
    print(
        "[interactions] state %d: %d interaction(s) drawn in group '%s'"
        % (state, total, group_name)
    )
    for itype in req_types:
        if counts.get(itype):
            print("    %-13s %d" % (itype, counts[itype]))
    if _active_engine[0] == "ds":
        if _parity_customized[0]:
            status = "customized"
        elif not _parity_source_integrity[0] or _last_parity_diagnostics:
            status = "degraded"
        else:
            status = "active"
        print(
            "    DockLens / Discovery Studio-like parity: %s (%s)"
            % (status, DSV_PARITY_CONTRACT)
        )
        for diagnostic in _last_parity_diagnostics:
            print("    parity note: %s" % diagnostic)
    if do_residues and res_seles:
        print("    interacting residues shown as sticks in '%s_residues'" % group_name)
    if not has_h:
        print(
            "    note: no hydrogens in structure -> H-bond angle checks "
            "skipped (heavy-atom distance mode). Add H for stricter results."
        )
    return counts


def _interaction_key(inter):
    """Stable identity of an interaction across trajectory frames."""
    return (inter["type"], inter["a_label"], inter["b_label"])


def interactions_occupancy(
    sel1="polymer",
    sel2="organic",
    types="all",
    start=1,
    end=0,
    threshold=0.0,
    draw=0,
    csv="",
    group_name="interactions",
):
    """Interaction persistence across an MD trajectory (multi-state object).

    Recomputes interactions for every state in [start, end] (end=0 -> last
    state) and reports, per unique interaction, the fraction of frames it is
    present ('occupancy', %). Prints a table sorted by occupancy.

    threshold  only report (and, if draw=1, redraw at the final state)
        interactions with occupancy >= this percentage.
    draw  1 => draw the interactions that pass `threshold` at state `end`.
    csv   path => also write the occupancy table to a CSV file.
    """
    _register_colors()
    sel1 = _resolve_receptor_selection(_resolve_selection(sel1))
    sel2 = _resolve_selection(sel2)
    req_types = _parse_types(types)
    start = int(start)
    available_states = max(
        int(cmd.count_states(sel1) or 1),
        int(cmd.count_states(sel2) or 1),
    )
    end = int(end) or available_states
    if start < 1 or end < start or end > available_states:
        raise ValueError(
            "State range must satisfy 1 <= start <= end <= %d"
            % available_states
        )
    nframes = end - start + 1
    if nframes > MAX_OCCUPANCY_FRAMES:
        raise ValueError(
            "Occupancy analysis is limited to %d frames per run"
            % MAX_OCCUPANCY_FRAMES
        )
    if int(draw):
        _validate_group_name(group_name)

    tally = {}  # key -> [count, sample_inter]
    total_compute_cost = 0
    for st in range(start, end + 1):
        inters, has_h = _compute_interactions(sel1, sel2, req_types, st)
        total_compute_cost += _last_compute_cost[0]
        if (
            st == start
            and nframes * _last_compute_cost[0]
            > MAX_OCCUPANCY_COMPUTE_COST
        ):
            raise ValueError(
                "Trajectory exceeds the combined processing budget "
                "(estimated %d operations; limit %d). Narrow selections "
                "or analyze fewer frames."
                % (
                    nframes * _last_compute_cost[0],
                    MAX_OCCUPANCY_COMPUTE_COST,
                )
            )
        if total_compute_cost > MAX_OCCUPANCY_COMPUTE_COST:
            raise ValueError(
                "Trajectory exceeds the combined processing budget "
                "(accumulated %d operations; limit %d). Narrow selections "
                "or analyze fewer frames."
                % (total_compute_cost, MAX_OCCUPANCY_COMPUTE_COST)
            )
        seen = set()
        for it in inters:
            key = _interaction_key(it)
            if key in seen:  # count a pair at most once per frame
                continue
            seen.add(key)
            if key not in tally:
                tally[key] = [0, it]
            tally[key][0] += 1

    rows = []
    for key, (cnt, sample) in tally.items():
        occ = 100.0 * cnt / nframes
        if occ >= float(threshold):
            rows.append((occ, cnt, sample))
    rows.sort(key=lambda r: r[0], reverse=True)

    print("=" * 68)
    print(
        " Interaction occupancy over states %d-%d (%d frames)" % (start, end, nframes)
    )
    print("=" * 68)
    print("  %6s  %-13s %s" % ("occ%", "type", "partners"))
    for occ, cnt, s in rows:
        print("  %6.1f  %-13s %s -- %s" % (occ, s["type"], s["a_label"], s["b_label"]))
    print("=" * 68)
    print(
        "  %d unique interaction(s) at threshold >= %.1f%%"
        % (len(rows), float(threshold))
    )

    if csv:
        _write_csv(
            csv,
            [
                "occupancy_pct",
                "frames_present",
                "n_frames",
                "type",
                "partner_a",
                "partner_b",
            ],
            [
                [round(occ, 2), cnt, nframes, s["type"], s["a_label"], s["b_label"]]
                for occ, cnt, s in rows
            ],
        )
        print("  occupancy table written to %s" % csv)

    if int(draw):
        # Redraw the full interaction set at the final state. Note: `threshold`
        # filters the printed/CSV table only, not which dashes are drawn.
        detect_interactions(
            sel1, sel2, types=req_types, state=end, group_name=group_name
        )
    return rows


def _write_csv(path, header, rows):
    """Minimal CSV writer (stdlib csv), used by occupancy + export."""
    import csv as _csvmod

    def safe_cell(value):
        if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
            return "'" + value
        return value

    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = _csvmod.writer(fh)
        w.writerow([safe_cell(value) for value in header])
        w.writerows([[safe_cell(value) for value in row] for row in rows])


def _normalized_interaction_endpoints(interaction):
    endpoints = {}
    for prefix in ("a", "b"):
        obj = interaction["%s_obj" % prefix]
        atom = _endpoint_atom(obj)
        side = getattr(atom, "side", "") or prefix
        endpoints[side] = {
            "label": interaction["%s_label" % prefix],
            "residue": atom.res_tag(),
            "atom": (
                interaction["%s_label" % prefix].rsplit("_", 1)[-1]
                if hasattr(obj, "atoms")
                else atom.name
            ),
            "serial": getattr(atom, "serial", None),
            "role": interaction.get("%s_role" % prefix, ""),
        }
    empty = {"label": "", "residue": "", "atom": "", "serial": None, "role": ""}
    return endpoints.get("receptor", empty), endpoints.get("ligand", empty)


def interactions_export_csv(
    filename, sel1="polymer", sel2="organic", types="all", state=1
):
    """Write all interactions for one state to a CSV file.

    The geometric fields are populated for face-directed interactions when
    explicit hydrogens are present; blank cells mean the metric is not used by
    that interaction class.
    """
    sel1 = _resolve_receptor_selection(_resolve_selection(sel1))
    sel2 = _resolve_selection(sel2)
    state = int(state)
    req_types = _parse_types(types)
    inters, has_h = _compute_interactions(sel1, sel2, req_types, state)
    if has_h is None:
        print("[interactions] empty selection; nothing exported.")
        return
    rows = []
    for interaction in inters:
        receptor, ligand = _normalized_interaction_endpoints(interaction)
        hydrogen = interaction.get("hydrogen_obj")
        rows.append(
            [
                state,
                _chemistry_profile_for_engine(),
                "ds_like" if _active_engine[0] == "ds" else "complete",
                DSV_PARITY_CONTRACT if _active_engine[0] == "ds" else "",
                _parity_is_active(),
                interaction["type"],
                interaction["subtype"],
                receptor["residue"],
                receptor["atom"],
                receptor["serial"],
                receptor["role"],
                ligand["residue"],
                ligand["atom"],
                ligand["serial"],
                ligand["role"],
                round(interaction["dist"], 2),
                interaction.get("chemistry_basis", ""),
                interaction.get("confidence", ""),
                hydrogen.label() if hydrogen is not None else "",
                getattr(hydrogen, "serial", "") if hydrogen is not None else "",
                _round_metric(interaction.get("hydrogen_acceptor_distance")),
                _round_metric(interaction.get("donor_hydrogen_acceptor_angle")),
                _round_metric(interaction.get("hydrogen_acceptor_base_angle")),
                _round_metric(interaction.get("theta")),
                _round_metric(interaction.get("receptor_water_distance")),
                _round_metric(interaction.get("ligand_water_distance")),
                _round_metric(interaction.get("water_angle")),
                interaction["a_label"],
                interaction["b_label"],
            ]
        )
    _write_csv(
        filename,
        [
            "state",
            "chemistry_profile",
            "analysis_profile",
            "parity_contract",
            "parity_active",
            "type",
            "subtype",
            "receptor_residue",
            "receptor_atom",
            "receptor_atom_serial",
            "receptor_role",
            "ligand_residue",
            "ligand_atom",
            "ligand_atom_serial",
            "ligand_role",
            "distance_A",
            "chemistry_basis",
            "chemistry_confidence",
            "hydrogen_atom",
            "hydrogen_atom_serial",
            "hydrogen_acceptor_distance_A",
            "donor_hydrogen_acceptor_angle_deg",
            "hydrogen_acceptor_base_angle_deg",
            "theta_deg",
            "receptor_water_distance_A",
            "ligand_water_distance_A",
            "water_angle_deg",
            "partner_a",
            "partner_b",
        ],
        rows,
    )
    print("[interactions] %d interaction(s) written to %s" % (len(rows), filename))
    return rows


def _round_metric(value):
    return "" if value is None else round(value, 2)


def interactions_figure_preset(ray=0, filename=""):
    """Apply publication-friendly display settings to the current scene.

    White background, subtle depth cueing off, thicker dashes already set per
    object. Optionally ray-traces and saves a PNG. Only changes runtime display
    settings for this session (never touches PyMOL config files).
    """
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 0)
    cmd.set("ray_shadows", 0)
    cmd.set("antialias", 2)
    cmd.set("dash_round_ends", 1)
    cmd.set("cartoon_transparency", 0.3)
    cmd.orient("interactions_residues or interactions")
    if int(ray):
        cmd.ray(1600, 1200)
    if filename:
        cmd.png(filename, dpi=300, ray=int(ray))
        print("[interactions] figure saved to %s" % filename)


def interactions_set_appearance(
    thickness=0.06, dash_scale=1.0, label_size=14, group_name="interactions"
):
    """Tune the appearance of the drawn interaction dashes.

    thickness   dash radius (line thickness) applied to every dash object.
    dash_scale  multiplier on each object's per-type dash length + gap, so the
                sandwich/T-shaped/dotted encoding is preserved while scaling.
    label_size  label text size for the interaction distance labels.
    """
    thickness = float(thickness)
    dash_scale = float(dash_scale)
    n = 0
    for name in list(_dash_base):
        base_l, base_g = _dash_base[name]
        cmd.set("dash_radius", thickness, name)
        cmd.set("dash_length", base_l * dash_scale, name)
        cmd.set("dash_gap", base_g * dash_scale, name)
        cmd.set("label_size", float(label_size), name)
        n += 1
    print(
        "[interactions] appearance applied to %d dash object(s) "
        "(thickness=%.3f, dash_scale=%.2f, label_size=%s)"
        % (n, thickness, dash_scale, label_size)
    )
    return n


def interactions_visibility(action="show", group_name="interactions"):
    """Show / hide / clear the interaction dashes, residue sticks and legend.

    action  'show'  -> enable all;  'hide' -> disable all;
            'clear' -> delete the group, residue selection, legend and helper.
    """
    action = str(action).strip().lower()
    group_name = _validate_group_name(group_name)
    targets = [group_name, "%s_residues" % group_name, "interactions_legend"]
    if action == "show":
        for t in targets:
            cmd.enable(t)
    elif action == "hide":
        for t in targets:
            cmd.disable(t)
    elif action == "clear":
        for t in targets + [_HELPER]:
            cmd.delete(t)
        _drawn_names.clear()
        _dash_base.clear()
    else:
        print("[interactions] visibility: action must be show|hide|clear")
        return
    print("[interactions] visibility '%s' applied." % action)


def interactions_set_engine(engine="ds"):
    """Switch the active detection engine: 'ds' (default) or 'plip'.

    Reloads CUTOFFS in place from CUTOFF_PROFILES[engine] and resets the
    per-engine defaults used by interactions_set_cutoff('reset', ...). Any
    cutoff values manually edited via interactions_set_cutoff or the "Edit
    cutoffs..." dialog under the previous engine are discarded on switch.
    """
    engine = str(engine).strip().lower()
    if engine not in CUTOFF_PROFILES:
        print(
            "[interactions] unknown engine '%s'. Valid: %s"
            % (engine, ", ".join(DETECTION_ENGINES))
        )
        return
    global _CUTOFF_DEFAULTS
    _active_engine[0] = engine
    _parity_customized[0] = False
    _last_parity_diagnostics[:] = []
    CUTOFFS.clear()
    CUTOFFS.update(CUTOFF_PROFILES[engine])
    _CUTOFF_DEFAULTS = dict(CUTOFFS)
    print("[interactions] detection engine set to '%s'." % engine)


def interactions_set_cutoff(key, value):
    """Set a single geometric cutoff at runtime (see the CUTOFFS table).

    Example:  interactions_set_cutoff pipi_dist, 5.0
    Pass key='reset' to restore all shipped defaults.
    """
    if str(key).strip().lower() == "reset":
        CUTOFFS.clear()
        CUTOFFS.update(_CUTOFF_DEFAULTS)
        _parity_customized[0] = False
        print("[interactions] all cutoffs reset to defaults.")
        return
    if key not in CUTOFFS:
        print(
            "[interactions] unknown cutoff '%s'. Valid keys: %s"
            % (key, ", ".join(sorted(CUTOFFS)))
        )
        return
    parsed = float(value)
    upper_bound = 180.0 if "angle" in key else 20.0
    if not math.isfinite(parsed) or not 0.0 <= parsed <= upper_bound:
        raise ValueError(
            "cutoff %s must be finite and between 0 and %.1f"
            % (key, upper_bound)
        )
    CUTOFFS[key] = parsed
    _parity_customized[0] = True
    print("[interactions] cutoff %s = %s" % (key, CUTOFFS[key]))
    if _active_engine[0] == "ds":
        print(
            "[interactions] custom cutoff active: DockLens parity is disabled "
            "until reset or engine re-selection."
        )


def interactions_parity_status():
    """Print and return the active DockLens parity contract state."""
    active = _parity_is_active()
    status = {
        "active": active,
        "engine": _active_engine[0],
        "contract": DSV_PARITY_CONTRACT,
        "customized": _parity_customized[0],
        "source_integrity": _parity_source_integrity[0],
        "diagnostics": tuple(_last_parity_diagnostics),
    }
    print(
        "[interactions] DockLens / Discovery Studio-like parity: %s (%s)"
        % ("active" if active else "inactive", DSV_PARITY_CONTRACT)
    )
    for diagnostic in _last_parity_diagnostics:
        print("    note: %s" % diagnostic)
    return status


def show_interaction_legend(onscreen=0, sele="all"):
    """Print the colour -> interaction-type legend to the PyMOL log.

    onscreen=1 also draws a 3D legend (one coloured, labelled marker per type)
    as a column beside `sele`'s bounding box, in a 'interactions_legend' object.
    Note: this is a 3D legend that rotates with the scene (PyMOL has no fixed
    2D overlay via cmd); place your final view before ray-tracing.
    """
    _register_colors()
    print("=" * 60)
    print(" Interaction legend (Okabe-Ito, colour-blind-safe)")
    print("=" * 60)
    for itype, (okabe_key, desc) in INTERACTION_COLORS.items():
        r, g, b = _OKABE_ITO[okabe_key]
        extra = "  (sandwich=long dash, T-shaped=short dash)" if itype == "pipi" else ""
        print(
            "  %-13s #%02X%02X%02X  %-11s %s%s"
            % (itype, r, g, b, okabe_key, desc, extra)
        )
    print("=" * 60)

    if not int(onscreen):
        return
    cmd.delete("interactions_legend")
    try:
        (x0, y0, z0), (x1, y1, z1) = cmd.get_extent(sele)
    except Exception:
        y0, x1, y1, z1 = 0, 10, 10, 10
    x = x1 + 2.0
    ytop = y1
    step = max(1.5, (y1 - y0) / (len(INTERACTION_COLORS) + 1))
    for i, itype in enumerate(INTERACTION_COLORS):
        yi = ytop - i * step
        aname = "leg_%d" % i
        cmd.pseudoatom(
            "interactions_legend",
            name=aname,
            pos=[x, yi, z1],
            label=itype,
        )
        cmd.color(_color_name(itype), "interactions_legend and name %s" % aname)
    cmd.set("label_size", 18, "interactions_legend")
    cmd.show("labels", "interactions_legend")
    cmd.show("nonbonded", "interactions_legend")
    print("  3D legend drawn in 'interactions_legend'")


# ===========================================================================
# Qt dialog (Plugin menu entry)
# ===========================================================================

_dialog = None  # module-level ref keeps the dialog from being GC'd

_GUI_MIN_SIZE = (320, 240)
_GUI_DEFAULT_SIZE = (580, 720)


def _dialog_sizes_for_screen(QtWidgets):
    """Return usable minimum and initial sizes for the active Qt screen.

    PyMOL can run with substantial high-DPI scaling, where a physically large
    display has a small logical height.  Cap both values to the available Qt
    work area so the scrollable dialog never starts beyond the visible screen.
    """
    app_class = getattr(QtWidgets, "QApplication", None)
    app = app_class.instance() if app_class is not None else None
    screen = app.primaryScreen() if app is not None else None
    if screen is None:
        return _GUI_MIN_SIZE, _GUI_DEFAULT_SIZE
    try:
        geometry = screen.availableGeometry()
        usable_width = max(1, int(geometry.width()) - 32)
        usable_height = max(1, int(geometry.height()) - 32)
    except Exception:
        return _GUI_MIN_SIZE, _GUI_DEFAULT_SIZE
    minimum = (
        min(_GUI_MIN_SIZE[0], usable_width),
        min(_GUI_MIN_SIZE[1], usable_height),
    )
    initial = (
        min(_GUI_DEFAULT_SIZE[0], usable_width),
        min(_GUI_DEFAULT_SIZE[1], usable_height),
    )
    return minimum, initial


def _build_scrollable_form(QtWidgets, QtCore, dialog):
    """Create the responsive, scrollable content area for the plug-in dialog.

    Keeping the complete form inside a ``QScrollArea`` means every control is
    reachable on compact laptop displays and under high-DPI scaling, while
    larger screens retain a comfortably sized resizable window.
    """
    minimum, initial = _dialog_sizes_for_screen(QtWidgets)
    dialog.setMinimumSize(*minimum)
    dialog.resize(*initial)
    dialog.setSizeGripEnabled(True)

    outer = QtWidgets.QVBoxLayout(dialog)
    outer.setContentsMargins(0, 0, 0, 0)
    scroll = QtWidgets.QScrollArea(dialog)
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
    scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)

    content = QtWidgets.QWidget(scroll)
    form = QtWidgets.QFormLayout(content)
    form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
    scroll.setWidget(content)
    outer.addWidget(scroll)
    return form


def run_plugin_gui():
    """Open the interaction-detector dialog (Plugin menu callback)."""
    global _dialog
    from pymol.Qt import QtCore, QtWidgets  # absent in headless PyMOL

    if _dialog is None:
        _dialog = QtWidgets.QDialog()
        _dialog.setWindowTitle("Non-Covalent Interactions %s" % PLUGIN_VERSION)
        form = _build_scrollable_form(QtWidgets, QtCore, _dialog)

        # Detection engine toggle (switch, like DockLens' engine switch).
        engine_box = QtWidgets.QGroupBox("Detection engine")
        ebox = QtWidgets.QHBoxLayout(engine_box)
        rb_plip = QtWidgets.QRadioButton("PLIP-style")
        rb_ds = QtWidgets.QRadioButton(
            "DockLens / Discovery Studio-like (recommended)"
        )
        rb_plip.setChecked(_active_engine[0] != "ds")
        rb_ds.setChecked(_active_engine[0] == "ds")
        ebox.addWidget(rb_plip)
        ebox.addWidget(rb_ds)
        form.addRow(engine_box)
        parity_label = QtWidgets.QLabel()
        parity_label.setWordWrap(True)

        def _update_parity_label():
            if _parity_is_active():
                text = "Parity active: %s" % DSV_PARITY_CONTRACT
            elif _active_engine[0] == "ds":
                text = "Parity degraded: check cutoffs and chemistry diagnostics"
            else:
                text = "PLIP mode: DockLens parity inactive"
            parity_label.setText(text)

        def _set_engine_from_gui(checked):
            interactions_set_engine("ds" if checked else "plip")
            _update_parity_label()

        rb_ds.toggled.connect(_set_engine_from_gui)
        _update_parity_label()
        form.addRow("Scientific profile:", parity_label)

        sel1 = QtWidgets.QLineEdit("polymer")
        sel2 = QtWidgets.QLineEdit("organic")
        form.addRow("Receptor (sel1):", sel1)
        form.addRow("Ligand (sel2):", sel2)

        # interaction-type checkboxes (all ticked by default)
        types_box = QtWidgets.QGroupBox("Interaction types")
        tgrid = QtWidgets.QGridLayout(types_box)
        checks = {}
        for i, itype in enumerate(VALID_TYPES):
            cb = QtWidgets.QCheckBox(itype)
            cb.setChecked(True)
            checks[itype] = cb
            tgrid.addWidget(cb, i // 2, i % 2)
        form.addRow(types_box)

        state = QtWidgets.QSpinBox()
        state.setMinimum(1)
        state.setMaximum(999999)
        state.setValue(1)
        form.addRow("State (MD frame):", state)

        auto_lig = QtWidgets.QCheckBox("Auto-detect ligand (ignore sel2)")
        disable_native = QtWidgets.QCheckBox("Hide PyMOL native H-bond dashes")
        disable_native.setChecked(True)
        keep_label = QtWidgets.QCheckBox("Keep distance labels")
        show_res = QtWidgets.QCheckBox("Show interacting residues as sticks")
        onscreen = QtWidgets.QCheckBox("On-screen 3D legend")
        form.addRow(auto_lig)
        form.addRow(disable_native)
        form.addRow(keep_label)
        form.addRow(show_res)
        form.addRow(onscreen)

        # Trajectory occupancy controls
        occ_box = QtWidgets.QGroupBox("Trajectory occupancy (MD)")
        ogrid = QtWidgets.QFormLayout(occ_box)
        occ_start = QtWidgets.QSpinBox()
        occ_start.setMinimum(1)
        occ_start.setMaximum(999999)
        occ_start.setValue(1)
        occ_end = QtWidgets.QSpinBox()
        occ_end.setMinimum(0)
        occ_end.setMaximum(999999)
        occ_end.setValue(0)
        occ_thr = QtWidgets.QDoubleSpinBox()
        occ_thr.setRange(0.0, 100.0)
        occ_thr.setValue(0.0)
        ogrid.addRow("Start state:", occ_start)
        ogrid.addRow("End state (0=last):", occ_end)
        ogrid.addRow("Min occupancy %:", occ_thr)
        form.addRow(occ_box)

        # Appearance controls (apply live to the drawn dashes)
        app_box = QtWidgets.QGroupBox("Appearance (live)")
        agrid = QtWidgets.QFormLayout(app_box)
        thick = QtWidgets.QDoubleSpinBox()
        thick.setRange(0.01, 0.50)
        thick.setSingleStep(0.01)
        thick.setValue(0.06)
        dscale = QtWidgets.QDoubleSpinBox()
        dscale.setRange(0.2, 4.0)
        dscale.setSingleStep(0.1)
        dscale.setValue(1.0)
        lsize = QtWidgets.QSpinBox()
        lsize.setRange(6, 40)
        lsize.setValue(14)
        agrid.addRow("Dash thickness:", thick)
        agrid.addRow("Dash length/gap scale:", dscale)
        agrid.addRow("Label size:", lsize)
        form.addRow(app_box)

        summary = QtWidgets.QLabel("No interactions drawn yet.")
        summary.setWordWrap(True)
        form.addRow(summary)

        btns = QtWidgets.QHBoxLayout()
        b_detect = QtWidgets.QPushButton("Detect")
        b_occ = QtWidgets.QPushButton("Occupancy")
        b_csv = QtWidgets.QPushButton("Export CSV")
        btns.addWidget(b_detect)
        btns.addWidget(b_occ)
        btns.addWidget(b_csv)
        form.addRow(btns)

        btns2 = QtWidgets.QHBoxLayout()
        b_show = QtWidgets.QPushButton("Show")
        b_hide = QtWidgets.QPushButton("Hide")
        b_clear = QtWidgets.QPushButton("Clear")
        btns2.addWidget(b_show)
        btns2.addWidget(b_hide)
        btns2.addWidget(b_clear)
        form.addRow(btns2)

        btns3 = QtWidgets.QHBoxLayout()
        b_legend = QtWidgets.QPushButton("Legend")
        b_fig = QtWidgets.QPushButton("Figure preset")
        b_adv = QtWidgets.QPushButton("Edit cutoffs...")
        b_close = QtWidgets.QPushButton("Close")
        btns3.addWidget(b_legend)
        btns3.addWidget(b_fig)
        btns3.addWidget(b_adv)
        btns3.addWidget(b_close)
        form.addRow(btns3)

        def _sels():
            s1 = sel1.text().strip() or "polymer"
            s2 = "auto" if auto_lig.isChecked() else (sel2.text().strip() or "organic")
            return s1, s2

        def _chosen():
            return [t for t, cb in checks.items() if cb.isChecked()]

        def _apply_appearance():
            interactions_set_appearance(
                thickness=thick.value(),
                dash_scale=dscale.value(),
                label_size=lsize.value(),
            )

        def _run():
            chosen = _chosen()
            if not chosen:
                return
            s1, s2 = _sels()
            counts = detect_interactions(
                s1,
                s2,
                types=chosen,
                state=state.value(),
                disable_native_hbond=1 if disable_native.isChecked() else 0,
                label=1 if keep_label.isChecked() else 0,
                show_residues=1 if show_res.isChecked() else 0,
            )
            _apply_appearance()  # honour current appearance settings
            if onscreen.isChecked():
                show_interaction_legend(onscreen=1)
            if counts:
                total = sum(counts.values())
                parts = ", ".join(
                    "%s=%d" % (t, counts[t]) for t in VALID_TYPES if counts.get(t)
                )
                summary.setText("%d interaction(s): %s" % (total, parts))
            else:
                summary.setText("No interactions found.")

        def _run_occ():
            chosen = _chosen()
            if not chosen:
                return
            s1, s2 = _sels()
            interactions_occupancy(
                s1,
                s2,
                types=chosen,
                start=occ_start.value(),
                end=occ_end.value(),
                threshold=occ_thr.value(),
            )

        def _run_csv():
            chosen = _chosen()
            if not chosen:
                return
            fn, _ = QtWidgets.QFileDialog.getSaveFileName(
                _dialog,
                "Export interactions CSV",
                "interactions.csv",
                "CSV files (*.csv)",
            )
            if not fn:
                return
            s1, s2 = _sels()
            interactions_export_csv(fn, s1, s2, types=chosen, state=state.value())

        b_detect.clicked.connect(_run)
        b_occ.clicked.connect(_run_occ)
        b_csv.clicked.connect(_run_csv)
        b_show.clicked.connect(lambda: interactions_visibility("show"))
        b_hide.clicked.connect(lambda: interactions_visibility("hide"))
        b_clear.clicked.connect(
            lambda: (interactions_visibility("clear"), summary.setText("Cleared."))
        )
        b_legend.clicked.connect(lambda: show_interaction_legend(onscreen=0))
        b_fig.clicked.connect(lambda: interactions_figure_preset())
        b_adv.clicked.connect(_open_cutoff_editor)
        b_close.clicked.connect(_dialog.hide)

        # live appearance updates
        thick.valueChanged.connect(_apply_appearance)
        dscale.valueChanged.connect(_apply_appearance)
        lsize.valueChanged.connect(_apply_appearance)

    _dialog.show()
    _dialog.raise_()


_cutoff_dialog = None


def _open_cutoff_editor():
    """Open a scrollable editor for the geometric CUTOFFS (Advanced)."""
    global _cutoff_dialog
    from pymol.Qt import QtWidgets

    if _cutoff_dialog is None:
        _cutoff_dialog = QtWidgets.QDialog()
        _cutoff_dialog.setWindowTitle("Edit interaction cutoffs")
        outer = QtWidgets.QVBoxLayout(_cutoff_dialog)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QtWidgets.QWidget()
        grid = QtWidgets.QFormLayout(inner)
        spins = {}
        for key in sorted(CUTOFFS):
            sp = QtWidgets.QDoubleSpinBox()
            sp.setDecimals(2)
            sp.setRange(0.0, 360.0)
            sp.setSingleStep(0.1)
            sp.setValue(float(CUTOFFS[key]))
            spins[key] = sp
            grid.addRow(key, sp)
        scroll.setWidget(inner)
        outer.addWidget(scroll)

        row = QtWidgets.QHBoxLayout()
        b_apply = QtWidgets.QPushButton("Apply")
        b_reset = QtWidgets.QPushButton("Reset defaults")
        b_done = QtWidgets.QPushButton("Close")
        row.addWidget(b_apply)
        row.addWidget(b_reset)
        row.addWidget(b_done)
        outer.addLayout(row)

        def _apply():
            for key, sp in spins.items():
                CUTOFFS[key] = float(sp.value())
            print("[interactions] cutoffs updated (re-run Detect to apply).")

        def _reset():
            interactions_set_cutoff("reset", 0)
            for key, sp in spins.items():
                sp.setValue(float(CUTOFFS[key]))

        b_apply.clicked.connect(_apply)
        b_reset.clicked.connect(_reset)
        b_done.clicked.connect(_cutoff_dialog.hide)
        _cutoff_dialog.resize(320, 480)
        _cutoff_dialog._spins = spins  # keep for refresh on re-show (engine switch)

    # Refresh displayed values from the active engine every time the dialog
    # is (re)opened, so an "Edit cutoffs..." click after switching engines
    # (or after a CLI interactions_set_cutoff call) shows current numbers.
    for key, sp in _cutoff_dialog._spins.items():
        sp.setValue(float(CUTOFFS[key]))

    _cutoff_dialog.show()
    _cutoff_dialog.raise_()


# ===========================================================================
# Registration
# ===========================================================================

cmd.extend("detect_interactions", detect_interactions)
cmd.extend("interactions_occupancy", interactions_occupancy)
cmd.extend("interactions_export_csv", interactions_export_csv)
cmd.extend("interactions_figure_preset", interactions_figure_preset)
cmd.extend("interactions_set_appearance", interactions_set_appearance)
cmd.extend("interactions_visibility", interactions_visibility)
cmd.extend("interactions_set_cutoff", interactions_set_cutoff)
cmd.extend("interactions_set_engine", interactions_set_engine)
cmd.extend("interactions_parity_status", interactions_parity_status)
cmd.extend("show_interaction_legend", show_interaction_legend)
cmd.extend("interactions_gui", run_plugin_gui)

cmd.auto_arg[0]["detect_interactions"] = [cmd.selection_sc, "selection", ", "]
cmd.auto_arg[1]["detect_interactions"] = [cmd.selection_sc, "selection", ", "]
cmd.auto_arg[0]["interactions_occupancy"] = [cmd.selection_sc, "selection", ", "]
cmd.auto_arg[1]["interactions_occupancy"] = [cmd.selection_sc, "selection", ", "]


def __init_plugin__(app=None):
    """PyMOL Plugin Manager entry point.

    Registers colours/commands and adds a 'Non-Covalent Interactions' item to
    the Plugin menu (this is what makes the plugin usable from the GUI, not just
    the command line).
    """
    _register_colors()
    try:
        from pymol.plugins import addmenuitemqt

        addmenuitemqt("Non-Covalent Interactions", run_plugin_gui)
    except Exception as exc:  # older PyMOL / no Qt: fall back to command only
        print(
            "[interactions] menu item not added (%s). "
            "Use the 'detect_interactions' command instead." % exc
        )
