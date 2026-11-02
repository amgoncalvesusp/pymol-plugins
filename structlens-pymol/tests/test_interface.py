import csv
import hashlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from structlens_pymol_plugin.interface import open_interface


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / "complex.pdb"
    source.write_bytes(b"END\n")
    report = tmp_path / "contacts.csv"
    header = "record source_sha256 model chain_a chain_b model_a model_b number_a insertion_a name_a number_b insertion_b name_b atom_a atom_b distance_angstrom cutoff_angstrom".split()
    with report.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, header)
        writer.writeheader()
        writer.writerow(dict(record="analysis", source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), model="0", chain_a="A", chain_b="B", cutoff_angstrom="5"))
        writer.writerow(dict(record="pair", chain_a="A", chain_b="B", model_a="0", model_b="0", number_a="'-1", insertion_a="A", number_b="3", name_a="ASP", name_b="LYS", atom_a="OD1", atom_b="NZ", distance_angstrom="3"))
    command = Mock()
    command.get_unused_name.side_effect = lambda prefix: prefix + "01"
    command.count_states.return_value = 1
    command.get_model.return_value.atom = [
        SimpleNamespace(chain=c, resi=r, resn=n, name=a, index=i)
        for c, r, n, a, i in [("A", "-1A", "ASP", "OD1", 2), ("B", "3", "LYS", "NZ", 5)]
    ]
    return command, report, source


def test_highlights_exact_indices(inputs):
    cmd, report, source = inputs
    open_interface(cmd, report, source)
    assert [call.args[2] for call in cmd.select_list.call_args_list] == [[2], [5]]


@pytest.mark.parametrize("problem", ["hash", "chain", "model", "missing", "multistate", "extension"])
def test_invalid_or_ambiguous_source(inputs, problem):
    cmd, report, source = inputs
    if problem == "hash":
        source.write_bytes(b"modified")
    elif problem == "extension":
        source = source.rename(source.with_suffix(".pml"))
    elif problem == "multistate":
        cmd.count_states.return_value = 2
    elif problem == "missing":
        cmd.get_model.return_value.atom = []
    else:
        row = report.read_text().splitlines()
        row[-1] = row[-1].replace("A,B,0,0", "Z,B,0,0" if problem == "chain" else "A,B,0,7")
        report.write_text("\n".join(row))
    with pytest.raises(ValueError):
        open_interface(cmd, report, source)
    cmd.show.assert_not_called()
def test_rejects_consistently_wrong_model_number(inputs):
    cmd, report, source = inputs
    report.write_text(report.read_text().replace(",0,A,B,", ",7,A,B,").replace(",A,B,0,0,", ",A,B,7,7,"))
    with pytest.raises(ValueError, match="model"):
        open_interface(cmd, report, source)
    cmd.load.assert_not_called()


def test_explicit_pdb_model_number(inputs):
    cmd, report, source = inputs
    old = hashlib.sha256(source.read_bytes()).hexdigest()
    source.write_bytes(b"MODEL        7\nENDMDL\nEND\n")
    report.write_text(report.read_text().replace(old, hashlib.sha256(source.read_bytes()).hexdigest()).replace(",0,A,B,", ",7,A,B,").replace(",A,B,0,0,", ",A,B,7,7,"))
    open_interface(cmd, report, source)
    assert cmd.select_list.call_count == 2
