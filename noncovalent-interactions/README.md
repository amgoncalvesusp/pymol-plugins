# PyMOL Non-Covalent Interactions Plugin

Detects and draws non-covalent **intermolecular** interactions (beyond PyMOL's
native H-bond dashes) as **individual, colour-coded dashed-line objects** for
docking-pose and MD-trajectory analysis. Each interaction becomes its own named
object in the side panel, ready for publication figures.

Reference scope: Discovery Studio Visualizer / LUNA, with geometric criteria
based on the open-source PLIP, Arpeggio and ProLIF families.

---

## Features

- **12 interaction types**, three geometric classes (atom–atom, atom–ring,
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

- **Colour-blind-safe** Okabe-Ito palette. `pipi` uses one colour with two dash
  styles: **sandwich = long dash**, **T-shaped = short dash**. The 4 extended
  types (>8) reuse a hue but are drawn **dotted** to stay distinct.
- **MD occupancy**: `interactions_occupancy` reports each interaction's
  persistence (% of frames) across a trajectory, ranked, with optional CSV.
- **CSV export**: `interactions_export_csv` dumps one state to CSV.
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

### Option A — Plugin Manager, single file (simplest)
1. `Plugin` ▸ `Plugin Manager` ▸ `Install New Plugin` ▸ `Choose file...`
2. Select **`interactions_plugin.py`**.
3. Restart PyMOL (or it loads immediately).

### Option B — Plugin Manager, zip package
1. `Plugin` ▸ `Plugin Manager` ▸ `Install New Plugin` ▸ `Choose file...`
2. Select **`pymol_interactions_plugin.zip`**.

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

The GUI dialog (`Plugin ▸ Non-Covalent Interactions`) exposes all of this: an
**Appearance** panel (live dash thickness / scale / label size), **Show / Hide /
Clear** buttons, a per-type **count summary** after Detect, and an **Edit
cutoffs...** advanced editor.

### Parameters

| Param | Default | Meaning |
|-------|---------|---------|
| `sel1` | `polymer` | Receptor side. Keep `sel1`/`sel2` as **distinct** groups. |
| `sel2` | `organic` | Ligand side. |
| `types` | `all` | Space/comma list, or `all`. Valid: `hbond carbon_hbond saltbridge pipi pication pialkyl alkyl halogen`. |
| `state` | `1` | Model state used for coordinates (MD frame number). |
| `disable_native_hbond` | `0` | `1` → hide PyMOL's own polar-contact/H-bond dashes. |
| `group_name` | `interactions` | Side-panel group collecting every object. |
| `label` | `0` | `1` → keep the numeric distance label on each dash. |
| `show_residues` | `0` | `1` → show interacting residues as sticks in `<group_name>_residues`. |

### Quick test

```python
run /path/to/interactions_plugin.py
fetch 1stp, async=0            # streptavidin + biotin (BTN)
detect_interactions polymer, organic
show_interaction_legend
```

---

## Notes & limitations

- **Cutoffs** live in the `CUTOFFS` dict at the top of `interactions_plugin.py`
  and are editable. Each carries a source comment; values without a firm
  literature consensus are flagged `UNCERTAIN` (`carbon_hbond`, `pialkyl`).
- **Hydrogens**: with explicit H present, H-bond/halogen **angle** checks apply.
  Without H, the plugin falls back to a heavy-atom distance-only mode and prints
  a note — add hydrogens for stricter results.
- **Ligand charge/aromaticity** relies on PDB `formal_charge` and ring
  planarity. Ligands lacking assigned formal charges won't yield salt-bridge or
  pi-cation contacts on that side; richer chemical typing would require RDKit
  (intentionally out of scope).
- Uses **only** the `pymol.cmd` API to draw and name objects.

---

## Files

- `interactions_plugin.py` — the plugin (single file, canonical).
- `pymol_interactions_plugin/` — Plugin Manager package wrapper.
- `pymol_interactions_plugin.zip` — packaged for zip install (Option B).
