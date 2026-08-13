# Changelog

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
