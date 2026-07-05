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

COMMANDS
  detect_interactions      detect + draw for one state (main command)
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
  types space/comma list, or 'all'. Valid (12):
        hbond carbon_hbond saltbridge pipi pication pialkyl alkyl halogen
        metal water_bridge pi_sulfur pi_anion
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

Geometric cutoffs follow PLIP defaults where available (Salentin et al.,
Nucleic Acids Res 2015; PLIP config.py). Values without a firm consensus are
marked "UNCERTAIN" in the CUTOFFS table below and are exposed as editable
module constants. Ring perception is topology-based (bond-graph cycles +
planarity heuristic) — no RDKit dependency.
"""

from __future__ import print_function

import numpy as np
from pymol import cmd


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
    "alkyl": ("blue", "Alkyl-alkyl (hydrophobic)"),
    "halogen": ("black", "Halogen bond"),
    # --- extended set (>8 types): colour is reused, drawn with a DOTTED dash so
    # it stays distinguishable from the 8 core types that share the same hue. ---
    "metal": ("vermillion", "Metal coordination (dotted)"),
    "water_bridge": ("skyblue", "Water-mediated H-bond (dotted)"),
    "pi_sulfur": ("reddishpurple", "pi-sulfur (dotted)"),
    "pi_anion": ("yellow", "pi-anion (dotted)"),
}

# Extended types are rendered dotted to disambiguate the reused hue.
_EXTENDED_TYPES = {"metal", "water_bridge", "pi_sulfur", "pi_anion"}


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
}
_PIPI_SANDWICH_DASH = (0.60, 0.20, 0.08)  # long dash
_PIPI_TSHAPED_DASH = (0.15, 0.45, 0.08)  # short dash


# ===========================================================================
# Geometric cutoffs  (edit here to override)
# ===========================================================================
#
# Sources:
#   PLIP    = Salentin et al. 2015, PLIP config.py defaults.
#   Steiner = Steiner, Angew. Chem. Int. Ed. 2002 (weak H-bonds).
#   DS      = Discovery Studio interaction definitions (proprietary; ranges
#             approximate, flagged UNCERTAIN).
#
CUTOFFS = {
    # Conventional H-bond: donor(N/O)...acceptor(N/O) heavy-atom distance.
    # PLIP HBOND_DIST_MAX = 4.1 A; D-H...A angle >= 100 deg (angle only if H present).
    "hbond_dist": 4.1,
    "hbond_angle": 100.0,
    # Carbon H-bond (weak): C...acceptor distance, C-H...A angle.
    # Steiner reports C...O up to ~3.5-4.0 A; angle typically >120 deg.
    # UNCERTAIN: no single agreed cutoff. Defaults chosen conservatively.
    "carbon_hbond_dist": 3.6,  # UNCERTAIN
    "carbon_hbond_angle": 120.0,  # UNCERTAIN
    # Salt bridge: distance between charged centres. PLIP SALTBRIDGE_DIST_MAX = 5.5.
    "saltbridge_dist": 5.5,
    # pi-pi stacking: centroid-centroid distance, planar offset, plane angle.
    # PLIP: PISTACK_DIST_MAX = 5.5, PISTACK_OFFSET_MAX = 2.0, PISTACK_ANG_DEV = 30.
    #   sandwich (parallel): inter-plane angle < 30 deg
    #   T-shaped (perpendicular): inter-plane angle in [60, 90] deg
    "pipi_dist": 5.5,
    "pipi_offset": 2.0,
    "pipi_angle_dev": 30.0,
    # pi-cation: cation...ring-centroid distance, planar offset.
    # PLIP PICATION_DIST_MAX = 6.0, offset <= 2.0.
    "pication_dist": 6.0,
    "pication_offset": 2.0,
    # pi-alkyl: ring-centroid...aliphatic-carbon distance.
    # DS-derived hydrophobic/pi contact ~4-6 A. UNCERTAIN.
    "pialkyl_dist": 5.0,  # UNCERTAIN
    # Alkyl-alkyl (hydrophobic C...C). PLIP HYDROPH_DIST_MAX = 4.0.
    "alkyl_dist": 4.0,
    # Halogen bond: X(Cl/Br/I)...acceptor(N/O/S) distance, C-X...A angle.
    # PLIP HALOGEN_DIST_MAX = 4.0; C-X...A angle 165 +/- 30 => >= 135 deg.
    "halogen_dist": 4.0,
    "halogen_angle": 135.0,
    # Metal coordination: metal ion...(O/N/S) distance. PLIP METAL_DIST_MAX = 3.0.
    "metal_dist": 3.0,
    # Water-mediated H-bond (bridge): each leg water-O...(donor/acceptor) heavy
    # distance, plus angle at the water O. PLIP WATER_BRIDGE_MINDIST = 2.5,
    # MAXDIST = 4.1, omega angle 75-140 deg.
    "water_bridge_min": 2.5,
    "water_bridge_max": 4.1,
    "water_bridge_angle_min": 75.0,
    "water_bridge_angle_max": 140.0,
    # pi-sulfur: aromatic-ring centroid...S distance.
    # Ringer et al. / Zauhar et al. report optimal ~5.3 A. UNCERTAIN (range 5.0-6.0).
    "pi_sulfur_dist": 5.3,  # UNCERTAIN
    # pi-anion: aromatic-ring centroid...anion distance + planar offset.
    # Less standardised; ~5.0 A above the ring plane. UNCERTAIN (range 4.5-5.0).
    "pi_anion_dist": 5.0,  # UNCERTAIN
    "pi_anion_offset": 2.0,
}

# Pristine copy of the shipped defaults (for the GUI "Reset" in the cutoff editor).
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

    def res_tag(self):
        chain = self.chain if self.chain else "_"
        return "%s%s%s" % (self.resn, self.resi, ("" if chain == "_" else chain))

    def label(self):
        return "%s_%s" % (self.res_tag(), self.name)

    def res_sele(self):
        """PyMOL selection string matching this atom's whole residue."""
        parts = ["resn %s" % self.resn, "resi \\%s" % self.resi]
        if self.chain:
            parts.append("chain %s" % self.chain)
        if self.segi:
            parts.append("segi %s" % self.segi)
        return "(" + " and ".join(parts) + ")"


def _load_atoms(selection, state):
    """Return (atoms, has_hydrogen) for a selection at a given state."""
    model = cmd.get_model(selection, state=state)
    atoms = [Atom(i, ca) for i, ca in enumerate(model.atom)]
    for bond in model.bond:
        i, j = bond.index
        atoms[i].neighbors.append(atoms[j])
        atoms[j].neighbors.append(atoms[i])
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


def _h_neighbors(atom):
    return [n for n in atom.neighbors if n.elem == "H"]


def classify(atoms, rings, has_h):
    """Return a dict of feature lists for one molecular side."""
    ring_atom_ids = set(a.idx for r in rings for a in r.atoms)

    donors = []  # (atom, [H atoms])  N/O-H donors
    carbon_donors = []  # (atom, [H atoms])  C-H donors
    acceptors = []  # atoms
    cations = []  # (point, label)
    anions = []  # (point, label)
    halogens = []  # (atom, bonded_C)  X-C
    alkyl_carbons = []  # sp3 aliphatic carbons
    metals = []  # metal ion atoms
    sulfurs = []  # S atoms (for pi-sulfur)

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
        pt = _centroid([x.coord for x in grp])
        lbl = "%s_guan" % res if resn == "ARG" else "%s_%s" % (res, grp[0].name)
        cations.append((pt, lbl, grp[0].res_sele()))
    for (res, _resn), grp in grouped_anion.items():
        pt = _centroid([x.coord for x in grp])
        anions.append((pt, "%s_carboxyl" % res, grp[0].res_sele()))

    # --- H-bond donors/acceptors, halogens, alkyl carbons ---
    for a in atoms:
        if a.elem in _HB_ACCEPTOR_ELEMS:
            # Exclude cationic N (e.g. ammonium/guanidinium) as acceptor.
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
                if hs:
                    carbon_donors.append((a, hs))
            # aliphatic carbon: not aromatic-ring member, bonded only to C/H
            if a.idx not in ring_atom_ids:
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

    return {
        "donors": donors,
        "carbon_donors": carbon_donors,
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
            if has_h and hs:
                best = max(_angle_at(h.coord, donor.coord, acc.coord) for h in hs)
                if best < angle_cut:
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
    for rings, alks in ((fa["rings"], fb["alkyl"]), (fb["rings"], fa["alkyl"])):
        for r in rings:
            for a in alks:
                if _dist(r.centroid, a.coord) <= cut:
                    out.append(
                        {
                            "type": "pialkyl",
                            "subtype": "",
                            "a_label": r.tag,
                            "b_label": a.label(),
                            "a_point": r.centroid,
                            "b_point": a.coord,
                            "a_sele": r.atoms[0].res_sele(),
                            "b_sele": a.res_sele(),
                        }
                    )
    return out


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
    for rings, sulfs in ((fa["rings"], fb["sulfurs"]), (fb["rings"], fa["sulfurs"])):
        for r in rings:
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
    # water_bridge handled separately (needs the water list) in _compute.
}


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


def _draw(interaction, group_name, keep_label):
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


def _load_waters(sel1, sel2, state):
    """Water-oxygen Atom objects near either selection (for water bridges)."""
    resn = "+".join(sorted(_WATER_RESN))
    near = "(resn %s) and elem O and (byres (all within 5 of ((%s) or (%s))))" % (
        resn,
        sel1,
        sel2,
    )
    if cmd.count_atoms(near) == 0:
        return []
    waters, _has_h = _load_atoms(near, state)
    return waters


def _compute_interactions(sel1, sel2, req_types, state):
    """Detect (but do NOT draw) all requested interactions for one state.

    Returns (interactions, has_h). Each interaction dict carries a computed
    'dist' (endpoint separation, Angstrom). Shared by detect_interactions,
    interactions_occupancy and interactions_export_csv.
    """
    atoms1, has_h1 = _load_atoms(sel1, state)
    atoms2, has_h2 = _load_atoms(sel2, state)
    if not atoms1 or not atoms2:
        return [], None
    has_h = has_h1 or has_h2

    feat1 = classify(atoms1, _build_rings(atoms1), has_h)
    feat2 = classify(atoms2, _build_rings(atoms2), has_h)

    inters = []
    for itype in req_types:
        if itype == "water_bridge":
            waters = _load_waters(sel1, sel2, state)
            inters.extend(detect_water_bridge(feat1, feat2, waters))
        else:
            inters.extend(_DETECTORS[itype](feat1, feat2, has_h))
    for it in inters:
        it["dist"] = _dist(it["a_point"], it["b_point"])
    return inters, has_h


def detect_interactions(
    sel1="polymer",
    sel2="organic",
    types="all",
    state=1,
    disable_native_hbond=0,
    group_name="interactions",
    label=0,
    show_residues=0,
):
    """Detect and draw non-covalent interactions between sel1 and sel2.

    See the module docstring for full parameter documentation and examples.
    Re-run with a different `state` to recompute a specific MD frame; the
    interaction group is rebuilt from scratch on every call.

    sel2='auto' picks the ligand automatically (organic, else non-polymer).
    show_residues 1 => also display interacting residues as sticks in a
        selection named '<group_name>_residues'.
    """
    _register_colors()
    sel1 = _resolve_selection(sel1)
    sel2 = _resolve_selection(sel2)
    state = int(state)
    keep_label = int(label) != 0
    do_disable = int(disable_native_hbond) != 0
    do_residues = int(show_residues) != 0
    req_types = _parse_types(types)

    # Fresh slate for this group + helper (idempotent re-run / per-frame redraw).
    cmd.delete(group_name)
    cmd.delete(_HELPER)
    cmd.delete("%s_residues" % group_name)
    _PSEUDO_COUNTER[0] = 0
    _drawn_names.clear()
    _dash_base.clear()

    if do_disable:
        _disable_native_hbonds()

    inters, has_h = _compute_interactions(sel1, sel2, req_types, state)
    if has_h is None:
        print("[interactions] empty selection (sel1 or sel2 has no atoms).")
        return

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
    sel1 = _resolve_selection(sel1)
    sel2 = _resolve_selection(sel2)
    req_types = _parse_types(types)
    start = int(start)
    end = int(end) or cmd.count_states(sel1) or 1
    if end < start:
        start, end = end, start
    nframes = end - start + 1

    tally = {}  # key -> [count, sample_inter]
    for st in range(start, end + 1):
        inters, has_h = _compute_interactions(sel1, sel2, req_types, st)
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

    with open(path, "w", newline="") as fh:
        w = _csvmod.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def interactions_export_csv(
    filename, sel1="polymer", sel2="organic", types="all", state=1
):
    """Write all interactions for one state to a CSV file.

    Columns: state, type, subtype, partner_a, partner_b, distance_A.
    """
    sel1 = _resolve_selection(sel1)
    sel2 = _resolve_selection(sel2)
    state = int(state)
    req_types = _parse_types(types)
    inters, has_h = _compute_interactions(sel1, sel2, req_types, state)
    if has_h is None:
        print("[interactions] empty selection; nothing exported.")
        return
    rows = [
        [
            state,
            it["type"],
            it["subtype"],
            it["a_label"],
            it["b_label"],
            round(it["dist"], 2),
        ]
        for it in inters
    ]
    _write_csv(
        filename,
        ["state", "type", "subtype", "partner_a", "partner_b", "distance_A"],
        rows,
    )
    print("[interactions] %d interaction(s) written to %s" % (len(rows), filename))
    return rows


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


def interactions_set_cutoff(key, value):
    """Set a single geometric cutoff at runtime (see the CUTOFFS table).

    Example:  interactions_set_cutoff pipi_dist, 5.0
    Pass key='reset' to restore all shipped defaults.
    """
    if str(key).strip().lower() == "reset":
        CUTOFFS.clear()
        CUTOFFS.update(_CUTOFF_DEFAULTS)
        print("[interactions] all cutoffs reset to defaults.")
        return
    if key not in CUTOFFS:
        print(
            "[interactions] unknown cutoff '%s'. Valid keys: %s"
            % (key, ", ".join(sorted(CUTOFFS)))
        )
        return
    CUTOFFS[key] = float(value)
    print("[interactions] cutoff %s = %s" % (key, CUTOFFS[key]))


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
        (x0, y0, z0), (x1, y1, z1) = (0, 0, 0), (10, 10, 10)
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


def run_plugin_gui():
    """Open the interaction-detector dialog (Plugin menu callback)."""
    global _dialog
    from pymol.Qt import QtWidgets  # imported lazily; absent in headless PyMOL

    if _dialog is None:
        _dialog = QtWidgets.QDialog()
        _dialog.setWindowTitle("Non-Covalent Interactions")
        form = QtWidgets.QFormLayout(_dialog)

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
