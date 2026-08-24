"""Build a deterministic Plugin Manager ZIP for StructLens-PyMOL."""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path


def build(version: str, root: Path) -> Path:
    source = root / "structlens_pymol_plugin"
    output_dir = root / "dist"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"StructLens-PyMOL-v{version}.zip"
    files: list[Path] = [root / "structlens_pymol_plugin.py", root / "README.md", root / "CHANGELOG.md", root / "THIRD_PARTY_NOTICES.md"]
    files.extend(
        path
        for path in source.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc"}
    )
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
            name = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).parents[1])
    args = parser.parse_args()
    output = build(args.version, args.root.resolve())
    checksum = output.with_suffix(output.suffix + ".sha256")
    checksum.write_text(f"{hashlib.sha256(output.read_bytes()).hexdigest()}  {output.name}\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
