# PyMOL Non-Covalent Interactions Plugin

Detects and draws non-covalent **intermolecular** interactions (beyond PyMOL's
native H-bond dashes) as **individual, colour-coded dashed-line objects** for
docking-pose and MD-trajectory analysis. Each interaction becomes its own named
object in the side panel, ready for publication figures.

The default analytical mode is the exact **DockLens DSV + Discovery
Studio-like parity contract** used by DockLens 1.0. The PyMOL figure, CSV and
trajectory occupancy therefore come from the same filtered interaction set as
DockLens. This does not claim to reproduce BIOVIA's proprietary implementation
internally.

Parity contract: `docklens-dsv-2026.07`.

---

## Features

- **15 interaction types**, three geometric classes (atom–atom, atom–ring,
  ring–ring):

  | Type | Class | Colour (Okabe-Ito) | Cutoff · source |
  |------|-------|--------------------|-----------------|
  | `hbond` | atom–atom | Sky-blue `#56B4E9` | 4.1 Å, ≥100° · PLIP |
  | `carbon_hbond` | atom–atom | Bluish-green `#009E73` | 3.6 Å, ≥120° · Steiner *(UNCERTAIN)* |
  | `saltbridge` | centre–centre | Vermillion `#D55E00` | 5.5 Å · PLIP |
  | `pipi` | ring–ring | Reddish-purple `#CC79A7` | 5.5 Å, offset 2.0, ang-dev 30° · PLIP |
  | `pication` | ring–centre | Yellow `#F0E442` | 6.0 Å · PLIP |
  | `pialkyl` | ring–atom | Orange `#E69F00` | 5.0 Å · DS *(UNCERTAIN)* |
  | `alkyl` | atom–atom | Blue `#0072B2` | 4.0 Å · PLIP |
  | `halogen` | atom–atom | Black `#000000` | 4.0 Å, ≥135° · PLIP |
  | `metal` | atom–atom | Vermillion (dotted) | 3.0 Å · PLIP |
  | `water_bridge` | atom–water–atom | Sky-blue (dotted) | 2.5–4.1 Å legs, 75–140° · PLIP |
  | `pi_sulfur` | ring–atom | Reddish-purple (dotted) | 5.3 Å · Ringer/Zauhar *(UNCERTAIN)* |
  | `pi_anion` | ring–centre | Yellow (dotted) | 5.0 Å, offset 2.0 *(UNCERTAIN)* |
  | `pi_sigma` | ring–C-H | Reddish-purple (dotted) | DS-calibrated face geometry |
  | `pi_donor_hbond` | ring–N/O-H | Bluish-green (dotted) | DS-calibrated face geometry |
  | `pi_lone_pair` | atom–ring | Sky-blue (dotted) | 3.5 Å, face angle ≤30° |

- **Two selectable detection engines** — a switch, not a merge: `ds`
  (default, exact DockLens parity) or `plip` (legacy Salentin et al. behavior).
  The DS mode uses the bundled DockLens core, its `dsv` chemistry profile and
  the final `ds_like` filter, including the 4.0 Å salt-bridge display limit.
  Its default `polymer` receptor scope is expanded internally to
  `polymer or metals`, matching DockLens for metalloproteins.
  Toggle with a
  radio button in the GUI, or
  `interactions_set_engine ds` / `detect_interactions ..., engine=ds` on the
  command line.
- **Colour-blind-safe** Okabe-Ito palette. `pipi` uses one colour with two dash
  styles: **sandwich = long dash**, **T-shaped = short dash**. The 4 extended
  types (>8) reuse a hue but are drawn **dotted** to stay distinct.
- **MD occupancy**: `interactions_occupancy` reports each interaction's
  persistence (% of frames) across a trajectory, ranked, with optional CSV.
- **CSV export**: `interactions_export_csv` dumps one state to CSV, including
  hydrogen, H-to-centroid distance and face-angle fields where applicable.
- **Responsive dialog**: controls stay available on compact or high-DPI screens
  through a resizable, screen-aware window with automatic scroll bars.
- **Publication preset**: `interactions_figure_preset` sets white bg, ray, PNG.
- **Auto-ligand**: pass `sel2=auto` to detect the ligand automatically.
- **3D on-screen legend**: `show_interaction_legend onscreen=1`.
- **Aromatic-ring perception is topology-based** (bond-graph 5/6-cycles +
  planarity heuristic) — **no RDKit / no external dependency**. Uses only
  `pymol.cmd` and `numpy`, both bundled with PyMOL.
- **Coexists with native H-bonds**: `disable_native_hbond=1` *hides* PyMOL's own
  polar-contact dashes (never deletes them) to avoid duplicate lines.
- **Static poses and MD trajectories**: pass a `state` to recompute a single
  frame; the interaction group is rebuilt on every call.
- Per-interaction naming, e.g.
  `LYS68_NZ--LIG_O1_saltbridge`, `PHE45_ring--LIG_ring_pipi`,
  `LIG_Cl1--ASP90_carboxyl_halogen`.

---

## Installation

### Option A — Plugin Manager, ZIP package (recommended)

1. `Plugin` ▸ `Plugin Manager` ▸ `Install New Plugin` ▸ `Choose file...`
2. Select **`pymol_interactions_plugin.zip`**.

The ZIP contains the reviewed DockLens core needed for scientific parity.

### Option B — repository checkout

1. `Plugin` ▸ `Plugin Manager` ▸ `Install New Plugin` ▸ `Choose file...`
2. Select **`interactions_plugin.py`**.
3. Keep the adjacent `pymol_interactions_plugin/` directory in place because
   the loader resolves the bundled core from that directory.

### Option C — no install, per-session
```python
run /path/to/interactions_plugin.py
```

All three register the commands `detect_interactions` and
`show_interaction_legend`.

---

## Usage

```python
show_interaction_legend                      # colour -> type table

# one docking pose (receptor = polymer, ligand = organic)
detect_interactions polymer, organic

# hide PyMOL's native polar contacts to avoid duplicate H-bond lines
detect_interactions polymer, organic, disable_native_hbond=1

# only some interaction types
detect_interactions polymer, resn LIG, types=hbond pipi saltbridge

# one MD-trajectory frame (state 10); re-run per frame to recompute
detect_interactions polymer, organic, state=10

# keep the numeric distance label on each dash
detect_interactions polymer, organic, label=1

# also show the interacting residues as sticks
detect_interactions polymer, organic, show_residues=1

# auto-detect the ligand instead of naming sel2
detect_interactions polymer, auto

# DockLens / Discovery Studio-like parity is active by default
detect_interactions polymer, organic
interactions_parity_status

# optional legacy PLIP mode
interactions_set_engine plip
detect_interactions polymer, organic, engine=plip

# MD trajectory: persistence (%) of each interaction over states 1..last
interactions_occupancy polymer, organic
interactions_occupancy polymer, organic, start=1, end=500, threshold=25
interactions_occupancy polymer, organic, csv=occupancy.csv

# dump one state's interactions to CSV
interactions_export_csv interactions.csv, polymer, organic

# publication view (white bg) + ray-traced PNG
interactions_figure_preset ray=1, filename=figure.png

# 3D on-screen colour legend
show_interaction_legend onscreen=1

# appearance: dash thickness, length/gap scale, label size (applies to drawn set)
interactions_set_appearance thickness=0.12, dash_scale=1.5, label_size=18

# show / hide / clear the dashes + residues + legend
interactions_visibility hide
interactions_visibility clear

# edit a cutoff at runtime (re-run detect to apply); 'reset' restores defaults
interactions_set_cutoff pipi_dist, 5.0
interactions_set_cutoff reset
```

The GUI dialog (`Plugin ▸ Non-Covalent Interactions`) exposes all of this: a
**Detection engine** radio-button toggle (PLIP-style / Discovery
Studio-style, switches immediately), an **Appearance** panel (live dash
thickness / scale / label size), **Show / Hide / Clear** buttons, a per-type
**count summary** after Detect, and an **Edit cutoffs...** advanced editor
(reflects whichever engine is currently active).

### Parameters

| Param | Default | Meaning |
|-------|---------|---------|
| `sel1` | `polymer` | Receptor side. In DS mode this default includes receptor metals. Keep `sel1`/`sel2` as **distinct** groups. |
| `sel2` | `organic` | Ligand side. |
| `types` | `all` | Space/comma list, or `all`. Valid: `hbond carbon_hbond saltbridge pipi pication pialkyl alkyl halogen metal water_bridge pi_sulfur pi_anion pi_sigma pi_donor_hbond pi_lone_pair`. |
| `state` | `1` | Model state used for coordinates (MD frame number). |
| `disable_native_hbond` | `1` | `1` → hide PyMOL's own polar-contact/H-bond dashes. |
| `group_name` | `interactions` | Side-panel group collecting every object. |
| `label` | `0` | `1` → keep the numeric distance label on each dash. |
| `show_residues` | `0` | `1` → show interacting residues as sticks in `<group_name>_residues`. |
| `engine` | `''` (keep current; initially `ds`) | `plip` or `ds` → switch detection engine before computing. |

### Quick test

```python
run /path/to/interactions_plugin.py
fetch 1stp, async=0            # streptavidin + biotin (BTN)
detect_interactions polymer, organic
show_interaction_legend
```

---

## Notes & limitations

- **Parity scope** means equality with DockLens' transparent
  `dsv + ds_like` interpretation. Discovery Studio Visualizer itself remains
  proprietary.
- **Cutoffs** may still be edited for exploratory work. Any edit marks the
  profile as custom and disables the parity declaration until `reset` or a new
  engine selection.
- **Hydrogens**: explicit-H structures use H···A, D–H···A and H···A–Y
  geometry. Without explicit H, the same conservative DockLens inferred-H
  fallback is used.
- **Chemical metadata**: MOL2/SYBYL types and bond orders are retained when
  PyMOL exposes them. The plugin reports a parity diagnostic when it must use
  the lower-information PDB fallback. For MOL2/PDBQT, PyMOL-inferred formal
  charges are ignored because DockLens treats those formats' charge columns as
  partial charges; PDB formal charges remain available.
- Uses **only** the `pymol.cmd` API to draw and name objects.

---

## Files

- `interactions_plugin.py` — repository loader; keep the package directory next
  to it.
- `pymol_interactions_plugin/` — Plugin Manager package wrapper.
- `pymol_interactions_plugin/docklens_core.py` — reviewed, hash-locked DockLens
  scientific core.
- `pymol_interactions_plugin.zip` — recommended Plugin Manager package.
