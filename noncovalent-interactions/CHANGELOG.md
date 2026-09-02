# Changelog

## 0.7.0 — 2026-09-01

### Fixed

- **No interactions were detected at all.** `_verify_reviewed_source`
  hashed raw file bytes, so a Windows checkout (Git rewrites LF to CRLF)
  failed the bundled-core integrity check and the plugin raised
  `ImportError` at import time. The digest is now computed over
  LF-normalised content, matching `_module_source_sha256`; modified content
  is still rejected.
- **Solvated pockets aborted before detection.** The per-frame compute
  budget charged each water the full receptor x ligand product; water
  screening is linear per side and is now costed accordingly (a 4000 x 40
  atom pocket with 150 waters went from 24.8M estimated operations, over
  the 10M limit, to 0.77M).
- **Crowded sites refused to draw.** Exceeding `MAX_DRAWN_INTERACTIONS`
  raised; the closest contacts are now drawn and the remainder reported.
- **The GUI Detect button could fail before detecting.** Appearance was
  applied first and aborted the run when `<group>_residues` did not exist
  yet. Detection runs first, and styling a missing target is skipped with a
  note instead of raising.
- **Nonbonded sphere size had no effect**: it targeted representation names
  (`nb_spheres`, `nonbonded`) instead of a selection.
- `interactions_figure_preset` no longer fails when the interaction group or
  residue selection is absent.
- Packaging and digest tests compared raw bytes and failed on any Windows
  checkout; they now compare reviewed content.

### Added

- `interactions_set_analysis_profile`: DockLens' reporting view (`complete`
  or `ds_like`) as an axis independent from the scientific profile, so the
  full DockLens mode matrix (plip/luna/dsv/luna_dsv x complete/ds_like) is
  reproducible. Reported by `interactions_parity_status` and in the CSV
  `analysis_profile` column, and selectable in the GUI.
- `interactions_best_angle`: orients the camera to the clearest view of the
  ligand-protein interaction network (PCA over interaction endpoints; the
  thinnest direction becomes the viewing axis, the ligand is kept on the
  camera side, framing includes the interacting residues). Reports a
  planarity score; `apply=0` reports without moving the camera. Available as
  the GUI's **Best angle** button.

### Compatibility

- The legacy `ds` engine name remains an alias for `dsv`.
- All existing commands, arguments and defaults are unchanged.

### Validation

- 87 plugin tests passing (13 new, covering each fixed defect and both new
  commands).
- Ruff, byte compilation, source/ZIP synchronization and package integrity
  checks passing.

## 0.6.0 — 2026-08-13

### Added

- PLIP, LUNA, DSV-like and conservative LUNA x DSV scientific profiles.
- All 18 DockLens interaction types, including chalcogen and charge families.
- Non-inflating default type sets and semantic pair deduplication.
- Expanded appearance controls for molecular colors, sticks, spheres, dashes,
  labels, transparency, hydrogens, background and ray rendering.
- Validated appearance input before any PyMOL scene mutation.
- Synchronized root/package sources and a rebuilt installable ZIP with SHA-256
  parity checks.

### Compatibility

- The legacy `ds` engine name remains accepted as an alias for `dsv`.
- Existing detection, occupancy and CSV commands remain available.

### Validation

- 74 plugin tests passing.
- Ruff, byte compilation, source/ZIP synchronization and package integrity
  checks passing.
