"""Display the contacting residues recorded by StructLens in its interface CSV."""

import csv
import hashlib
import io
import math
import tempfile
from collections import defaultdict
from pathlib import Path


def _read(path, limit):
    with Path(path).open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"{path} exceeds the supported file size.")
    return data


def _identity(row, key):
    value = row[key]
    if value is None:
        raise ValueError("Truncated interface CSV row.")
    if value.startswith("'") and value[1:].lstrip().startswith(("=", "+", "-", "@")):
        return value[1:]
    return value


def _contact_rows(path):
    data = _read(path, 100 * 1024**2).decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(data, newline=""), strict=True)
    scope = next(reader, None)
    sides = ("a", "b")
    required = {"record", "source_sha256", "model", "cutoff_angstrom", "distance_angstrom"}
    required.update(f"{key}_{side}" for side in sides for key in ("chain", "model", "number", "insertion", "name", "atom"))
    if not scope or scope.get("record") != "analysis" or not required.issubset(scope):
        raise ValueError("Expected a StructLens subunit interface CSV.")
    chains = tuple(_identity(scope, "chain_" + side) for side in sides)
    cutoff = float(scope["cutoff_angstrom"])
    if chains[0] == chains[1] or not scope["model"] or not math.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("Invalid chain, model or cutoff in interface CSV.")
    contacts = (set(), set())
    for row in reader:
        if None in row or None in row.values():
            raise ValueError("Malformed interface CSV row.")
        if row["record"] in {"interaction", "residue"}:
            continue
        if row["record"] != "pair" or not 0 <= float(row["distance_angstrom"]) <= cutoff:
            raise ValueError("Invalid interface pair or distance.")
        for side, chain, atoms in zip(sides, chains, contacts):
            def field(key):
                return _identity(row, f"{key}_{side}")
            if field("model") != scope["model"] or field("chain") != chain:
                raise ValueError("Residue outside declared model or chain.")
            atoms.add((chain, field("number") + field("insertion"), field("name"), field("atom")))
    if not contacts[0]:
        raise ValueError("No contacting residues in this interface report.")
    return scope["source_sha256"], scope["model"], contacts


def _source_models(data, fmt):
    if fmt == "pdb":
        return {str(int(line[10:14])) for line in data.splitlines() if line.startswith(b"MODEL ")} or {"0"}
    from chempy.cif import parse_cif, unquote

    for block in parse_cif(data.decode("utf-8-sig")):
        for loop in block.loops:
            if "_atom_site_id" not in loop.keys:
                continue
            index = loop.get_col_idx_opt("_atom_site.pdbx_pdb_model_num")
            # Bio.PDB uses model id -1 for an mmCIF atom table without a serial column.
            return {"-1"} if index < 0 else {str(int(unquote(row[index]))) for row in loop.rows}
    raise ValueError("Missing mmCIF atom table for model verification.")


def open_interface(command, csv_path, source_path):
    """Load verified coordinates and highlight recorded contacting residues."""
    digest, model, contacts = _contact_rows(csv_path)
    suffix = Path(source_path).suffix.lower()
    formats = {".pdb": "pdb", ".ent": "pdb", ".cif": "cif", ".mmcif": "cif"}
    if suffix not in formats:
        raise ValueError("Choose the original uncompressed PDB or mmCIF file.")
    data = _read(source_path, 256 * 1024**2)
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("Coordinates do not match the source SHA-256 in the report.")
    if _source_models(data, formats[suffix]) != {model}:
        raise ValueError("CSV model does not identify the sole model in the original coordinates.")
    obj = command.get_unused_name("sl_interface")
    owned = [obj]
    try:
        with tempfile.TemporaryDirectory() as tmp:
            verified = Path(tmp) / ("complex" + suffix)
            verified.write_bytes(data)
            command.load(str(verified), obj, format=formats[suffix], multiplex=0, zoom=0)
        if command.count_states(obj) != 1:
            raise ValueError("Only single-model coordinates are supported for interface display.")
        residues = defaultdict(list)
        atom_names = set()
        for atom in command.get_model(obj, state=1).atom:
            residue = (atom.chain, atom.resi, atom.resn)
            residues[residue].append(atom.index)
            atom_names.add((*residue, atom.name))
        for side, atoms, color in zip(("a", "b"), contacts, ("cyan", "orange")):
            if not atoms.issubset(atom_names):
                raise ValueError("Reported contact atom missing in PyMOL coordinates.")
            indices = sorted(index for residue in {key[:3] for key in atoms} for index in residues[residue])
            selection = command.get_unused_name("sl_interface_" + side)
            owned.append(selection)
            command.select_list(selection, obj, indices, mode="index")
            command.show("sticks", selection)
            command.color(color, selection)
        command.zoom(" or ".join(owned[1:]))
    except Exception:
        for name in reversed(owned):
            command.delete(name)
        raise
    return obj
