# PyMOL Non-Covalent Interactions Plugin

Version 0.7.1 detects and draws intermolecular non-covalent interactions as
individual, color-coded PyMOL distance objects for docking poses and
multi-state trajectories.

## Release 0.7.1

This patch release applies occupancy thresholds to drawn contacts, rejects
invalid thresholds before touching a PyMOL session, and protects user objects
from group-name collisions. Sources and the Plugin Manager ZIP are synchronized.

## Release 0.7.0

Bug-fix and feature release: detection works again on Windows installs,
solvated pockets and crowded sites, DockLens' second mode axis (analysis
view) is selectable, and a best-viewing-angle command was added.

- **Fixed - no interactions were detected at all.** The bundled-core
  integrity check hashed raw bytes, so a Windows checkout (Git rewrites LF
  to CRLF) failed verification and the plugin raised `ImportError` on
  import. The digest is now taken over LF-normalised content, which is what
  the reviewed-source guarantee actually means; tampered content is still
  rejected.
- **Fixed - solvated pockets returned nothing.** The per-frame budget
  charged every water the full receptor x ligand product, so an ordinary
  structure with crystallographic waters exceeded the limit and aborted.
  Water screening is linear in each side, and is now costed that way.
- **Fixed - crowded sites refused to draw.** Passing the drawing limit
  raised instead of drawing; the closest contacts are now drawn and the
  remainder reported.
- **Fixed - the GUI Detect button could die before detecting.** Appearance
  was applied first and aborted the run when the interacting-residue
  selection did not exist yet. Detection now runs first and styling of a
  missing target is skipped instead of raising.
- **Fixed - nonbonded sphere size did nothing**, because it targeted
  representation names instead of a selection.
- **Added - `interactions_set_analysis_profile`**: DockLens' reporting view
  (`complete` or `ds_like`) as an independent axis from the scientific
  profile, so every DockLens mode combination is reproducible.
- **Added - `interactions_best_angle`**: orients the camera to the clearest
  view of the ligand-protein interaction network.

## Release 0.6.0

This release synchronized the PyMOL plugin with DockLens 1.1.0 and the
shared scientific interaction contract.

- Four canonical profiles are available: legacy PLIP, LUNA 0.14, corrected
  DSV-like and conservative LUNA x DSV.
- The hybrid profile combines representable native families with stricter
  shared geometry, semantic deduplication and no salt-bridge/charge double
  counting.
- All 18 canonical interaction types and the DockLens color contract are
  available in the detector, legend, CSV export and GUI.
- The GUI and commands expose protein/ligand/residue colors, stick radius,
  nonbonded sphere size, dash thickness and scale, labels, transparency,
  background, hydrogens and safe ray-render settings.
- Appearance arguments are validated before PyMOL mutations; profile changes,
  occupancy and CSV metadata report the active scientific contract.
- The installable ZIP contains the synchronized reviewed core and exact
  SHA-256 parity checks.

See [CHANGELOG.md](CHANGELOG.md) for the complete release record.

The plugin bundles the reviewed DockLens scientific core and locks it by
SHA-256 over its LF-normalised content. Its current contract is
`docklens-scientific-profiles-2026.08`. PyMOL detection, drawing, occupancy,
and CSV export use the native DockLens detector criteria; the `ds_like`
reporting filter is available on demand through the analysis view below.

## Analysis views (DockLens mode matrix)

The scientific profile and the reporting view are independent axes, which
together reproduce every DockLens mode:

- `complete` (default): keep every record the profile detects.
- `ds_like`: DockLens' Discovery Studio reporting filter (DS-reported
  families only, salt bridges beyond the DS distance ceiling dropped).

```python
interactions_set_engine luna_dsv          # chemistry profile
interactions_set_analysis_profile ds_like  # reporting view
detect_interactions polymer, organic
```

The active view is reported by `interactions_parity_status` and written to
the `analysis_profile` column of every CSV export.

## Best viewing angle

After a detection run, `interactions_best_angle` orients the camera to the
clearest view of the interaction network:

```python
detect_interactions polymer, organic, show_residues=1
interactions_best_angle
interactions_best_angle buffer=3.0
interactions_best_angle apply=0     # report the view without moving
```

It runs a principal-component analysis over the interaction endpoints: the
two directions of largest spread become the screen plane and the thinnest
direction becomes the viewing axis, so dashes spread out instead of
stacking behind one another. The ligand is kept on the camera side of the
receptor, the view is framed on the ligand plus interacting residues, and
the reported `planarity` (0-1) says how well the network fits one plane.
It is also available as the GUI's **Best angle** button.

## Scientific profiles

The profile selector exposes exactly four canonical profiles:

- `plip`: legacy PLIP-oriented thresholds and interaction families.
- `luna`: LUNA-oriented thresholds and non-inflating default families.
- `dsv`: Discovery Studio Visualizer-calibrated DockLens profile and default.
- `luna_dsv`: conservative LUNA/DSV hybrid.

`ds` remains accepted as a compatibility alias for `dsv`, but it is not
listed as a separate profile.

```python
interactions_set_engine dsv
interactions_set_engine luna
detect_interactions polymer, organic, engine=luna_dsv
interactions_parity_status
```

The active `CUTOFFS` table is an exact copy of the selected bundled-core
snapshot. Editing a cutoff marks the profile custom and disables the locked
contract status until `interactions_set_cutoff reset` or profile reselection.

### Non-inflating `types=all`

`types=all` means "the active profile's defaults", not unconditionally every
known family. The GUI initializes and refreshes its checkboxes from the same
`default_types_for_profile` snapshot.

- `plip` defaults to the legacy families and excludes the three newly gated
  families.
- `luna` defaults to its supported subset.
- `dsv` and `luna_dsv` default to all 18 canonical families.

An explicit type list may select any canonical family under any profile:

```python
detect_interactions polymer, resn LIG, types=hbond chalcogen charge_repulsion
```

The 18 canonical types are:
`hbond`, `carbon_hbond`, `saltbridge`, `attractive_charge`,
`charge_repulsion`, `pipi`, `pication`, `pialkyl`, `pi_sigma`,
`alkyl`, `halogen`, `metal`, `water_bridge`, `pi_sulfur`,
`pi_anion`, `pi_donor_hbond`, `pi_lone_pair`, and `chalcogen`.

Extended families reuse the color-blind-safe Okabe-Ito palette and use dotted
patterns so they remain visually distinct. In particular,
`attractive_charge`, `charge_repulsion`, and `chalcogen` are dotted.
Pi-pi contacts retain long dashes for sandwich and short dashes for T-shaped
geometry.

## Installation

Recommended PyMOL Plugin Manager installation:

1. Open `Plugin > Plugin Manager > Install New Plugin > Choose file...`.
2. Select `pymol_interactions_plugin.zip`.

The archive contains exactly:

- `pymol_interactions_plugin/__init__.py`
- `pymol_interactions_plugin/interactions_plugin.py`
- `pymol_interactions_plugin/docklens_core.py`
- `pymol_interactions_plugin/docklens_analysis_profiles.py`

For a repository checkout, keep the package directory adjacent to the root
`interactions_plugin.py`, then install/run that root file:

```python
run /path/to/interactions_plugin.py
```

## Detection and export

```python
# Active default is dsv; types=all resolves to dsv defaults
detect_interactions polymer, organic

# Auto-detect the ligand
detect_interactions polymer, auto

# One trajectory state and selected families
detect_interactions polymer, organic, types=hbond pipi saltbridge, state=10

# Keep labels and show participating residues as sticks
detect_interactions polymer, organic, label=1, show_residues=1

# Occupancy over a trajectory
interactions_occupancy polymer, organic, start=1, end=500, threshold=25
interactions_occupancy polymer, organic, csv=occupancy.csv

# Native-profile metadata is included in the CSV
interactions_export_csv interactions.csv, polymer, organic

# Visibility and legend
show_interaction_legend onscreen=1
interactions_visibility hide
interactions_visibility show
interactions_visibility clear
```

With `interactions_occupancy ..., threshold=75, draw=1`, the table and drawing
both apply the occupancy threshold (a finite percentage from 0 to 100).
The drawing uses only qualifying contacts present in the final analyzed state;
contacts absent from that state remain in the table but are not drawn.

Detection and clearing refuse a `group_name` already occupied by a molecule
or a group not created by this loaded plugin instance. Choose a new group name
when retaining groups restored from an earlier session.

The default `polymer` receptor is expanded to `polymer or metals` for
`dsv` and `luna_dsv`, matching the bundled detector scope for
metalloproteins. Receptor and ligand selections must remain distinct.

## Appearance API

Old calls remain valid:

```python
interactions_set_appearance thickness=0.12, dash_scale=1.5, label_size=18
```

The expanded command can style molecules, participating residues, interaction
objects, labels, hydrogens, background, and safe render settings in one call:

```python
interactions_set_appearance \
    protein_selection="polymer", \
    ligand_selection="resn LIG", \
    protein_color="gray70", \
    ligand_color="orange", \
    interacting_residue_color="marine", \
    stick_radius=0.22, \
    nonbond_sphere_size=0.31, \
    dash_thickness=0.09, \
    dash_scale=1.5, \
    label_size=18, \
    cartoon_transparency=0.25, \
    background_color="white", \
    show_hydrogens=1, \
    transparency=0.15, \
    ambient=0.35, \
    specular=0.2, \
    ray_shadows=0, \
    ray_opaque_background=0, \
    antialias=2
```

`thickness` remains the legacy alias/positional argument for
`dash_thickness`. `nonbond_sphere_scale` is accepted as a compatibility
alias for `nonbond_sphere_size`.

All appearance arguments are validated before the first mutating PyMOL call.
Selections reject command separators and control characters. Colors must be a
safe built-in name, `gray00` through `gray100` (or `grey`), a registered
`ii_*` interaction color, or a `#RRGGBB`/`0xRRGGBB` literal. Numeric
settings must be finite and within safe ranges.

The GUI exposes the same profile, per-type, molecule, dash, label, hydrogen,
transparency, background, and render controls. Appearance changes are applied
with the explicit **Apply appearance** button; detection reapplies the selected
settings to newly rebuilt dash objects.

## Publication preset

```python
interactions_figure_preset ray=1, filename=figure.png
```

This applies a white-background publication preset and can save a 300 DPI PNG.

## Scientific notes

- Explicit-hydrogen structures use the native profile's H...A, D-H...A, and
  acceptor-base geometry. Structures without hydrogen use DockLens' conservative
  inferred-hydrogen fallback where the selected profile permits it.
- MOL2/SYBYL atom types, bond orders, formal charges, and partial charges are
  retained when PyMOL exposes them. Lower-information inputs can produce
  diagnostics in `interactions_parity_status`.
- Ring perception is topology/planarity based and requires no RDKit dependency.
- Disabling native H-bonds hides matching PyMOL objects; it never deletes them.

## Files

- `interactions_plugin.py`: repository loader and command implementation.
- `pymol_interactions_plugin/interactions_plugin.py`: byte-identical packaged
  implementation.
- `pymol_interactions_plugin/docklens_core.py`: mechanically synchronized,
  hash-locked DockLens core.
- `pymol_interactions_plugin/docklens_analysis_profiles.py`: synchronized
  DockLens analysis helpers with the minimal bundled-module import adapter.
- `pymol_interactions_plugin.zip`: exact four-file Plugin Manager package.
