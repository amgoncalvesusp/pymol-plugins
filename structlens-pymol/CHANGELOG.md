# Changelog

## 0.3.1 — 2026-09-06

- Load mmCIF with the correct parser and remove temporary coordinate files.
- Apply stored transforms only to targets, using PyMOL's matrix layout.
- Scope selections to the imported objects and active target; preserve user objects.
- Render valid displacement CGO objects and use existing selection names for focus.
- Keep the previous analysis when opening a new bundle fails.
- Register the menu through the PyMOL plugin API.

## 0.3.0

- Added read-only MSA/conservation, interaction, site, vector, and residue
  evidence bundle sections.
- Added conserved/gained/lost interaction selections and site selections.
- Added displacement-vector rendering from authoritative bundle coordinates;
  no PyMOL-side scientific recalculation is performed.
- Added v0.3 compatibility tests and release ZIP/checksum naming.

## 0.1.0

- Initial StructLens-PyMOL plugin skeleton.
- Safe `.structlens-pymol` reader with schema-major compatibility checks.
- Namespaced structure loading, transforms, semantic selections, and presets.
- `structlens_open` command registration and Plugin Manager release packaging.
