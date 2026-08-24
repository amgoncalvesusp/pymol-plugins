# Changelog

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
