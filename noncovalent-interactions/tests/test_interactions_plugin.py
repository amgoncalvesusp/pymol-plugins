"""Regression tests for the Discovery Studio-calibrated PyMOL detectors.

The plug-in normally runs inside PyMOL.  A tiny command stub lets its pure
geometry and feature code be tested in a regular Python environment.
"""

import importlib.util
import importlib
import hashlib
import os
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest


def _load_plugin():
    pymol = types.ModuleType("pymol")
    pymol.cmd = types.SimpleNamespace(
        extend=lambda *_args, **_kwargs: None,
        selection_sc=object(),
        auto_arg=[{}, {}, {}, {}],
    )
    sys.modules["pymol"] = pymol
    path = Path(__file__).parents[1] / "interactions_plugin.py"
    spec = importlib.util.spec_from_file_location("docklens_pymol_plugin", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plugin = _load_plugin()


class _FakeDialog:
    def __init__(self):
        self.minimum_size = None
        self.size = None
        self.size_grip_enabled = False
        self.layout = None

    def setMinimumSize(self, width, height):
        self.minimum_size = (width, height)

    def resize(self, width, height):
        self.size = (width, height)

    def setSizeGripEnabled(self, enabled):
        self.size_grip_enabled = enabled


class _FakeLayout:
    AllNonFixedFieldsGrow = "grow"

    def __init__(self, parent):
        self.parent = parent
        self.widgets = []
        self.margins = None
        self.field_growth_policy = None
        parent.layout = self

    def setContentsMargins(self, *margins):
        self.margins = margins

    def addWidget(self, widget):
        self.widgets.append(widget)

    def setFieldGrowthPolicy(self, policy):
        self.field_growth_policy = policy


class _FakeScrollArea:
    def __init__(self, parent):
        self.parent = parent
        self.resizable = False
        self.horizontal_policy = None
        self.vertical_policy = None
        self.widget = None

    def setWidgetResizable(self, value):
        self.resizable = value

    def setHorizontalScrollBarPolicy(self, policy):
        self.horizontal_policy = policy

    def setVerticalScrollBarPolicy(self, policy):
        self.vertical_policy = policy

    def setWidget(self, widget):
        self.widget = widget


class _FakeWidget:
    def __init__(self, parent):
        self.parent = parent
        self.layout = None


class _FakeQtWidgets:
    QVBoxLayout = _FakeLayout
    QFormLayout = _FakeLayout
    QScrollArea = _FakeScrollArea
    QWidget = _FakeWidget


class _FakeQtCore:
    class Qt:
        ScrollBarAsNeeded = "as-needed"


class _FakeScreen:
    def availableGeometry(self):
        return types.SimpleNamespace(width=lambda: 360, height=lambda: 384)


class _FakeApplication:
    @staticmethod
    def instance():
        return types.SimpleNamespace(primaryScreen=lambda: _FakeScreen())


class _FakeQtWidgetsWithScreen(_FakeQtWidgets):
    QApplication = _FakeApplication


def test_gui_layout_is_scrollable_and_resizable_for_small_screens():
    dialog = _FakeDialog()
    form = plugin._build_scrollable_form(_FakeQtWidgets, _FakeQtCore, dialog)

    scroll = dialog.layout.widgets[0]
    assert dialog.minimum_size == (320, 240)
    assert dialog.size == (580, 720)
    assert dialog.size_grip_enabled is True
    assert scroll.resizable is True
    assert scroll.horizontal_policy == "as-needed"
    assert scroll.vertical_policy == "as-needed"
    assert scroll.widget.layout is form


def test_gui_size_is_capped_by_available_high_dpi_screen():
    minimum, initial = plugin._dialog_sizes_for_screen(_FakeQtWidgetsWithScreen)

    assert minimum == (320, 240)
    assert initial == (328, 352)


def test_gui_entrypoint_creates_a_real_scroll_area(monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtCore = pytest.importorskip("PyQt5.QtCore")
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    pymol_qt = types.ModuleType("pymol.Qt")
    pymol_qt.QtCore = QtCore
    pymol_qt.QtWidgets = QtWidgets
    monkeypatch.setitem(sys.modules, "pymol.Qt", pymol_qt)

    plugin._dialog = None
    plugin.run_plugin_gui()
    scroll = plugin._dialog.findChild(QtWidgets.QScrollArea)

    assert app is not None
    assert scroll is not None
    assert scroll.widgetResizable() is True
    plugin._dialog.close()
    plugin._dialog.deleteLater()
    plugin._dialog = None


def _atom(
    idx,
    elem,
    coord,
    name,
    *,
    sybyl_type="",
    resn="LIG",
    fcharge=0,
):
    atom = object.__new__(plugin.Atom)
    atom.idx = idx
    atom.elem = elem
    atom.name = name
    atom.resn = resn
    atom.resi = "1"
    atom.chain = ""
    atom.segi = ""
    atom.coord = np.asarray(coord, dtype=float)
    atom.fcharge = fcharge
    atom.neighbors = []
    atom.sybyl_type = sybyl_type
    atom.partial_charge = None
    atom.bond_orders = {}
    return atom


def _bond(left, right, order="1"):
    left.neighbors.append(right)
    right.neighbors.append(left)
    left.bond_orders[right.idx] = order
    right.bond_orders[left.idx] = order


def _ring():
    atoms = [
        _atom(10 + i, "C", point, "C%d" % i)
        for i, point in enumerate(
            ((1, 0, 0), (0.5, 0.866, 0), (-0.5, 0.866, 0), (-1, 0, 0),
             (-0.5, -0.866, 0), (0.5, -0.866, 0))
        )
    ]
    return plugin.Ring(atoms, "PHE34_ring")


def test_ds_profile_matches_calibrated_geometry():
    plip = plugin.CUTOFF_PROFILES["plip"]
    ds = plugin.CUTOFF_PROFILES["ds"]
    assert plugin.DSV_PARITY_CONTRACT == "docklens-dsv-2026.07"
    assert {
        key: value
        for key, value in plip.items()
        if key not in {"pi_lone_pair_dist", "pi_lone_pair_angle"}
    } == dict(plugin._docklens_core.cutoffs_for_preset("plip"))
    assert plip["pi_lone_pair_dist"] == 3.5
    assert plip["pi_lone_pair_angle"] == 30.0
    assert ds == dict(plugin._docklens_core.cutoffs_for_preset("dsv"))
    assert plugin.VALID_TYPES == plugin._docklens_core.VALID_TYPES
    assert plugin.INTERACTION_COLORS == plugin._docklens_core.INTERACTION_COLORS
    assert ds["hbond_dist"] == 4.1
    assert ds["hbond_h_a_dist"] == 3.1
    assert ds["hbond_acceptor_angle"] == 90.0
    assert ds["hbond_inferred_dist"] == 3.5
    assert ds["carbon_hbond_h_a_dist"] == 3.0
    assert ds["pialkyl_dist"] == 4.9
    assert ds["alkyl_dist"] == 4.2
    assert ds["metal_dist"] == 3.0
    assert ds["pi_sigma_h_centroid_dist"] == 4.3
    assert ds["pi_donor_h_centroid_dist"] == 4.1
    assert ds["pi_lone_pair_dist"] == 3.5
    assert ds["pi_lone_pair_angle"] == 30.0
    assert {"pi_sigma", "pi_donor_hbond", "pi_lone_pair"} <= set(plugin.VALID_TYPES)


def test_bundled_core_is_the_reviewed_docklens_core():
    path = (
        Path(__file__).parents[1]
        / "pymol_interactions_plugin"
        / "docklens_core.py"
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    assert digest == "dfdee432587c26f5cbd1025ecd110d160966b805083100aec2832970ae99af1b"


def test_bundled_analysis_profile_copy_is_present():
    path = (
        Path(__file__).parents[1]
        / "pymol_interactions_plugin"
        / "docklens_analysis_profiles.py"
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    assert digest == "8f33342682f5ae5568ae7acc728f04584695d26facba22ff271cb8bf80341823"


def test_unreviewed_fallback_module_is_rejected_before_import(tmp_path):
    path = tmp_path / "docklens_core.py"
    path.write_text("raise RuntimeError('must never execute')\n", encoding="utf-8")

    with pytest.raises(ImportError, match="integrity"):
        plugin._verify_reviewed_source(path, "0" * 64)


def test_compute_budget_accounts_for_water_bridge_cartesian_work():
    assert plugin._estimate_compute_cost(10, 20, 3, include_water=True) == 890
    assert plugin._estimate_compute_cost(10, 20, 3, include_water=False) == 200


def test_root_and_plugin_manager_sources_are_identical():
    root = Path(__file__).parents[1]
    assert (root / "interactions_plugin.py").read_bytes() == (
        root / "pymol_interactions_plugin" / "interactions_plugin.py"
    ).read_bytes()


def test_plugin_manager_package_imports_the_bundled_core():
    packaged = importlib.import_module(
        "pymol_interactions_plugin.interactions_plugin"
    )

    assert packaged.DSV_PARITY_CONTRACT == plugin.DSV_PARITY_CONTRACT
    assert {
        key: value
        for key, value in packaged.CUTOFF_PROFILES["plip"].items()
        if key not in {"pi_lone_pair_dist", "pi_lone_pair_angle"}
    } == dict(packaged._docklens_core.cutoffs_for_preset("plip"))
    assert packaged.CUTOFF_PROFILES["ds"] == dict(
        packaged._docklens_core.cutoffs_for_preset("dsv")
    )
    assert (
        packaged._docklens_analysis_profiles._DS_LIKE_TYPES
        == plugin._docklens_analysis_profiles._DS_LIKE_TYPES
    )


def test_installable_zip_contains_the_exact_reviewed_sources():
    root = Path(__file__).parents[1]
    expected = {
        "pymol_interactions_plugin/__init__.py",
        "pymol_interactions_plugin/interactions_plugin.py",
        "pymol_interactions_plugin/docklens_core.py",
        "pymol_interactions_plugin/docklens_analysis_profiles.py",
    }
    with zipfile.ZipFile(root / "pymol_interactions_plugin.zip") as archive:
        assert set(archive.namelist()) == expected
        for name in (
            "__init__.py",
            "interactions_plugin.py",
            "docklens_core.py",
            "docklens_analysis_profiles.py",
        ):
            assert archive.read("pymol_interactions_plugin/" + name) == (
                root / "pymol_interactions_plugin" / name
            ).read_bytes()


def test_ds_is_the_default_engine_for_docklens_figure_parity():
    assert plugin._active_engine[0] == "ds"


def test_ds_default_receptor_selection_includes_receptor_metals():
    plugin.interactions_set_engine("ds")

    assert plugin._resolve_receptor_selection("polymer") == "(polymer or metals)"

    plugin.interactions_set_engine("plip")
    assert plugin._resolve_receptor_selection("polymer") == "polymer"
    plugin.interactions_set_engine("ds")


def test_pymol_model_loader_preserves_docklens_chemistry_fields(monkeypatch):
    atoms = [
        types.SimpleNamespace(
            symbol="N",
            name="N1",
            resn="LIG",
            resi="1",
            chain="",
            segi="",
            coord=(0.0, 0.0, 0.0),
            formal_charge=0,
            partial_charge=-0.321,
            text_type="N.am",
            model="ligand_pose",
            index=41,
        ),
        types.SimpleNamespace(
            symbol="C",
            name="C1",
            resn="LIG",
            resi="1",
            chain="",
            segi="",
            coord=(1.3, 0.0, 0.0),
            formal_charge=0,
            partial_charge=0.321,
            text_type="C.2",
            model="ligand_pose",
            index=42,
        ),
    ]
    model = types.SimpleNamespace(
        atom=atoms,
        bond=[types.SimpleNamespace(index=(0, 1), order=2)],
    )
    monkeypatch.setattr(
        plugin.cmd, "get_model", lambda *_args, **_kwargs: model, raising=False
    )

    loaded, has_hydrogen = plugin._load_atoms("ligand", 1)

    assert has_hydrogen is False
    assert loaded[0].sybyl_type == "N.am"
    assert loaded[0].partial_charge == pytest.approx(-0.321)
    assert loaded[0].bond_orders[loaded[1].idx] == "2"
    assert loaded[0].res_sele() == '(byres (model "ligand_pose" and index 41))'


def test_mol2_sybyl_atoms_ignore_pymol_inferred_formal_charge():
    catom = types.SimpleNamespace(
        symbol="O",
        name="O1",
        resn="LIG",
        resi="1",
        chain="",
        segi="",
        coord=(0.0, 0.0, 0.0),
        formal_charge=-1,
        partial_charge=-0.5,
        text_type="O.co2",
        model="ligand",
        index=1,
    )

    atom = plugin.Atom(0, catom)

    assert atom.fcharge == 0
    assert atom.sybyl_type == "O.co2"


def test_pdb_placeholder_type_preserves_explicit_formal_charge():
    catom = types.SimpleNamespace(
        symbol="Zn",
        name="ZN",
        resn="ZN",
        resi="301",
        chain="A",
        segi="",
        coord=(0.0, 0.0, 0.0),
        formal_charge=2,
        partial_charge=0.0,
        text_type="??",
        model="receptor",
        index=1,
    )

    atom = plugin.Atom(0, catom)

    assert atom.fcharge == 2


def test_ds_chemistry_excludes_amide_acceptor_and_carbonyl_donor():
    amide_nitrogen = _atom(
        1, "N", (0, 0, 0), "N1", sybyl_type="N.am"
    )
    carbonyl_oxygen = _atom(
        2, "O", (3, 0, 0), "O1", sybyl_type="O.2"
    )
    hydrogen = _atom(3, "H", (4, 0, 0), "H1")
    _bond(carbonyl_oxygen, hydrogen)

    features = plugin.classify(
        [amide_nitrogen, carbonyl_oxygen, hydrogen],
        [],
        has_h=True,
        chemistry_profile="dsv",
    )

    assert amide_nitrogen not in features["acceptors"]
    assert features["donors"] == []


def test_ds_explicit_hbond_matches_docklens_three_geometry_filter():
    plugin.interactions_set_engine("ds")
    donor = _atom(1, "N", (0, 0, 0), "N", sybyl_type="N.am")
    hydrogen = _atom(2, "H", (1, 0, 0), "HN")
    acceptor = _atom(3, "O", (3.9, 0, 0), "O", sybyl_type="O.2")
    acceptor_base = _atom(4, "C", (3.9, 1, 0), "C", sybyl_type="C.2")
    _bond(donor, hydrogen)
    _bond(acceptor, acceptor_base, "2")
    donor_features = plugin.classify(
        [donor, hydrogen], [], has_h=True, chemistry_profile="dsv"
    )
    acceptor_features = plugin.classify(
        [acceptor, acceptor_base], [], has_h=True, chemistry_profile="dsv"
    )

    records = plugin.detect_hbond(donor_features, acceptor_features, True)

    assert len(records) == 1
    record = records[0]
    assert record["hydrogen"] == "LIG1_HN"
    assert record["chemistry_basis"] == "explicit_hydrogen"
    assert record["hydrogen_acceptor_distance_A"] == pytest.approx(2.9)
    assert record["donor_hydrogen_acceptor_angle_deg"] == pytest.approx(180.0)
    assert record["hydrogen_acceptor_base_angle_deg"] == pytest.approx(90.0)


def test_ds_explicit_hbond_rejects_missing_or_invalid_acceptor_base():
    plugin.interactions_set_engine("ds")
    donor = _atom(1, "N", (0, 0, 0), "N", sybyl_type="N.am")
    hydrogen = _atom(2, "H", (1, 0, 0), "HN")
    isolated_acceptor = _atom(3, "O", (3, 0, 0), "O", sybyl_type="O.2")
    _bond(donor, hydrogen)
    donor_features = plugin.classify(
        [donor, hydrogen], [], has_h=True, chemistry_profile="dsv"
    )
    isolated_features = plugin.classify(
        [isolated_acceptor], [], has_h=True, chemistry_profile="dsv"
    )

    assert plugin.detect_hbond(donor_features, isolated_features, True) == []

    acceptor_base = _atom(4, "C", (2, 0, 0), "C", sybyl_type="C.2")
    _bond(isolated_acceptor, acceptor_base, "2")
    invalid_features = plugin.classify(
        [isolated_acceptor, acceptor_base],
        [],
        has_h=True,
        chemistry_profile="dsv",
    )
    assert plugin.detect_hbond(donor_features, invalid_features, True) == []


def test_ds_emits_one_auditable_hbond_record_per_qualifying_hydrogen():
    plugin.interactions_set_engine("ds")
    donor = _atom(1, "N", (0, 0, 0), "ND2", sybyl_type="N.am")
    hydrogen_1 = _atom(2, "H", (1, 0.1, 0), "HD21")
    hydrogen_2 = _atom(3, "H", (1, -0.1, 0), "HD22")
    acceptor = _atom(4, "O", (3.5, 0, 0), "O", sybyl_type="O.2")
    acceptor_base = _atom(5, "C", (3.5, 0, 1), "C", sybyl_type="C.2")
    _bond(donor, hydrogen_1)
    _bond(donor, hydrogen_2)
    _bond(acceptor, acceptor_base, "2")

    records = plugin.detect_hbond(
        plugin.classify(
            [donor, hydrogen_1, hydrogen_2],
            [],
            has_h=True,
            chemistry_profile="dsv",
        ),
        plugin.classify(
            [acceptor, acceptor_base],
            [],
            has_h=True,
            chemistry_profile="dsv",
        ),
        True,
    )

    assert {record["hydrogen"] for record in records} == {
        "LIG1_HD21",
        "LIG1_HD22",
    }
    assert {record["a_label"] for record in records} == {
        "LIG1_HD21",
        "LIG1_HD22",
    }
    assert {
        tuple(np.asarray(record["a_point"]).round(3)) for record in records
    } == {
        tuple(hydrogen_1.coord.round(3)),
        tuple(hydrogen_2.coord.round(3)),
    }


def test_ds_carbon_hbond_requires_a_polarized_carbon():
    unpolarized = _atom(1, "C", (0, 0, 0), "CB", sybyl_type="C.3")
    carbon_neighbor = _atom(2, "C", (-1, 0, 0), "CA", sybyl_type="C.3")
    unpolarized_h = _atom(3, "H", (1, 0, 0), "HB")
    polarized = _atom(4, "C", (0, 4, 0), "CA", sybyl_type="C.3")
    nitrogen_neighbor = _atom(5, "N", (-1, 4, 0), "N", sybyl_type="N.am")
    polarized_h = _atom(6, "H", (1, 4, 0), "HA")
    _bond(unpolarized, carbon_neighbor)
    _bond(unpolarized, unpolarized_h)
    _bond(polarized, nitrogen_neighbor)
    _bond(polarized, polarized_h)

    features = plugin.classify(
        [
            unpolarized,
            carbon_neighbor,
            unpolarized_h,
            polarized,
            nitrogen_neighbor,
            polarized_h,
        ],
        [],
        has_h=True,
        chemistry_profile="dsv",
    )

    assert [atom for atom, _hydrogens in features["carbon_donors"]] == [polarized]


def test_ds_alkyl_feature_matches_docklens_protein_sidechain_policy():
    histidine_cb = _atom(
        1, "C", (0, 0, 0), "CB", sybyl_type="C.3", resn="HIS"
    )
    histidine_ca = _atom(
        2, "C", (1.5, 0, 0), "CA", sybyl_type="C.3", resn="HIS"
    )
    cysteine_cb = _atom(
        3, "C", (4, 0, 0), "CB", sybyl_type="C.3", resn="CYS"
    )
    cysteine_sg = _atom(
        4, "S", (5.5, 0, 0), "SG", sybyl_type="S.3", resn="CYS"
    )
    _bond(histidine_cb, histidine_ca)
    _bond(cysteine_cb, cysteine_sg)

    features = plugin.classify(
        [histidine_cb, histidine_ca, cysteine_cb, cysteine_sg],
        [],
        has_h=False,
        chemistry_profile="dsv",
    )

    assert features["alkyl"] == [cysteine_cb]


def test_ds_pi_lone_pair_requires_aromatic_evidence():
    plugin.interactions_set_engine("ds")
    aliphatic_atoms = [
        _atom(
            10 + index,
            "C",
            (
                float(np.cos(index * np.pi / 3)),
                float(np.sin(index * np.pi / 3)),
                0.0,
            ),
            "C%d" % index,
            sybyl_type="C.3",
        )
        for index in range(6)
    ]
    for index, atom in enumerate(aliphatic_atoms):
        _bond(atom, aliphatic_atoms[(index + 1) % 6], "1")
    ring = plugin.Ring(aliphatic_atoms, "LIG1_ring")
    acceptor = _atom(20, "O", (0, 0, 3.0), "O", sybyl_type="O.2")

    records = plugin.detect_pi_lone_pair(
        {"acceptors": [acceptor], "rings": []},
        {"acceptors": [], "rings": [ring]},
    )

    assert records == []


def test_compute_pipeline_applies_docklens_ds_like_saltbridge_filter(monkeypatch):
    cation = _atom(1, "N", (0, 0, 0), "NQ", sybyl_type="N.4", fcharge=1)

    monkeypatch.setattr(
        plugin,
        "_load_atoms",
        lambda selection, _state, index_offset=0: (
            ([cation], False)
            if selection == "receptor"
            else (
                [
                    _atom(
                        index_offset,
                        "O",
                        (4.1, 0, 0),
                        "O1",
                        sybyl_type="O.co2",
                        fcharge=-1,
                    )
                ],
                False,
            )
        ),
    )
    plugin.interactions_set_engine("ds")
    filtered, _has_h = plugin._compute_interactions(
        "receptor", "ligand", ["saltbridge"], 1
    )

    assert filtered == []

    plugin.interactions_set_engine("plip")
    complete, _has_h = plugin._compute_interactions(
        "receptor", "ligand", ["saltbridge"], 1
    )
    assert len(complete) == 1
    plugin.interactions_set_engine("ds")


def test_compute_pipeline_assigns_distinct_indices_to_receptor_and_ligand(
    monkeypatch,
):
    donor = _atom(0, "N", (0, 0, 0), "N", sybyl_type="N.am")
    hydrogen = _atom(1, "H", (1, 0, 0), "HN")
    acceptor = _atom(0, "O", (3, 0, 0), "O", sybyl_type="O.2")
    acceptor_base = _atom(1, "C", (3, 1, 0), "C", sybyl_type="C.2")
    _bond(donor, hydrogen)
    _bond(acceptor, acceptor_base, "2")
    monkeypatch.setattr(
        plugin,
        "_load_atoms",
        lambda selection, _state, index_offset=0: _indexed_hbond_side(
            selection,
            index_offset,
        ),
    )
    plugin.interactions_set_engine("ds")

    records, _has_h = plugin._compute_interactions(
        "receptor", "ligand", ["hbond"], 1
    )

    assert len(records) == 1
    assert records[0]["hydrogen"] == "LIG1_HN"


def _indexed_hbond_side(selection, index_offset):
    if selection == "receptor":
        donor = _atom(index_offset, "N", (0, 0, 0), "N", sybyl_type="N.am")
        hydrogen = _atom(index_offset + 1, "H", (1, 0, 0), "HN")
        _bond(donor, hydrogen)
        return [donor, hydrogen], True
    acceptor = _atom(index_offset, "O", (3, 0, 0), "O", sybyl_type="O.2")
    acceptor_base = _atom(
        index_offset + 1,
        "C",
        (3, 1, 0),
        "C",
        sybyl_type="C.2",
    )
    _bond(acceptor, acceptor_base, "2")
    return [acceptor, acceptor_base], False


def test_compute_pipeline_returns_one_semantic_water_bridge(monkeypatch):
    receptor = _atom(
        1,
        "O",
        (3.0, 0.0, 0.0),
        "OG",
        resn="SER",
    )
    ligand = _atom(
        2,
        "O",
        (-1.5, 2.598076, 0.0),
        "O1",
        sybyl_type="O.2",
    )
    ligand_base = _atom(
        3,
        "C",
        (-2.5, 2.598076, 0.0),
        "C1",
        sybyl_type="C.2",
    )
    water = _atom(4, "O", (0.0, 0.0, 0.0), "O", resn="HOH")
    _bond(ligand, ligand_base, "2")

    monkeypatch.setattr(
        plugin,
        "_load_atoms",
        lambda selection, _state, index_offset=0: (
            ([receptor], False)
            if selection == "receptor"
            else ([ligand, ligand_base], False)
        ),
    )
    monkeypatch.setattr(
        plugin,
        "_load_waters",
        lambda *_args, **_kwargs: [water],
    )
    plugin.interactions_set_engine("ds")

    records, _has_h = plugin._compute_interactions(
        "receptor", "ligand", ["water_bridge"], 1
    )

    assert len(records) == 1
    assert records[0]["type"] == "water_bridge"
    assert records[0]["water_obj"] is water


def test_water_bridge_draws_two_legs_but_remains_one_record(monkeypatch):
    partner_a = _atom(1, "O", (3, 0, 0), "OG", resn="SER")
    partner_b = _atom(2, "O", (-1.5, 2.598076, 0), "O1")
    water = _atom(3, "O", (0, 0, 0), "O", resn="HOH")
    distances = []
    monkeypatch.setattr(
        plugin,
        "_pseudo_at",
        lambda point: "point_%s" % (tuple(np.asarray(point).round(3)),),
    )
    for name, value in {
        "distance": lambda name, start, end: distances.append((name, start, end)),
        "set": lambda *_args, **_kwargs: None,
        "hide": lambda *_args, **_kwargs: None,
        "group": lambda *_args, **_kwargs: None,
    }.items():
        monkeypatch.setattr(plugin.cmd, name, value, raising=False)
    plugin._drawn_names.clear()

    result = plugin._draw(
        {
            "type": "water_bridge",
            "subtype": "",
            "a_label": partner_a.label(),
            "b_label": partner_b.label(),
            "a_point": partner_a.coord,
            "b_point": partner_b.coord,
            "a_sele": partner_a.res_sele(),
            "b_sele": partner_b.res_sele(),
            "water_obj": water,
        },
        "interactions",
        False,
    )

    assert len(result) == 2
    assert len(distances) == 2


def test_csv_uses_normalized_docklens_endpoints_and_full_geometry(monkeypatch):
    receptor = _atom(1, "N", (0, 0, 0), "N", resn="ASN")
    ligand = _atom(2, "O", (3, 0, 0), "O1")
    hydrogen = _atom(3, "H", (1, 0, 0), "HN", resn="ASN")
    receptor.side = "receptor"
    ligand.side = "ligand"
    hydrogen.side = "receptor"
    receptor.serial = 11
    ligand.serial = 22
    hydrogen.serial = 12
    interaction = {
        "type": "hbond",
        "subtype": "",
        "a_label": receptor.label(),
        "b_label": ligand.label(),
        "a_point": receptor.coord,
        "b_point": ligand.coord,
        "a_obj": receptor,
        "b_obj": ligand,
        "a_role": "donor",
        "b_role": "acceptor",
        "dist": 3.0,
        "chemistry_basis": "explicit_hydrogen",
        "confidence": "high",
        "hydrogen_obj": hydrogen,
        "hydrogen_acceptor_distance": 2.0,
        "donor_hydrogen_acceptor_angle": 180.0,
        "hydrogen_acceptor_base_angle": 90.0,
    }
    captured = {}
    monkeypatch.setattr(
        plugin,
        "_compute_interactions",
        lambda *_args: ([interaction], True),
    )
    monkeypatch.setattr(
        plugin,
        "_write_csv",
        lambda _path, header, rows: captured.update(header=header, rows=rows),
    )
    plugin.interactions_set_engine("ds")

    plugin.interactions_export_csv("ignored.csv", "receptor", "ligand")

    row = dict(zip(captured["header"], captured["rows"][0]))
    assert row["parity_contract"] == plugin.DSV_PARITY_CONTRACT
    assert row["parity_active"] is True
    assert row["receptor_residue"] == "ASN1"
    assert row["receptor_atom"] == "N"
    assert row["ligand_residue"] == "LIG1"
    assert row["ligand_atom"] == "O1"
    assert row["hydrogen_atom"] == "ASN1_HN"
    assert row["hydrogen_acceptor_distance_A"] == 2.0
    assert row["donor_hydrogen_acceptor_angle_deg"] == 180.0
    assert row["hydrogen_acceptor_base_angle_deg"] == 90.0


def test_custom_cutoff_disables_parity_until_reset():
    plugin.interactions_set_engine("ds")
    assert plugin.interactions_parity_status()["active"] is True


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, 181.0])
def test_invalid_angle_cutoffs_are_rejected(value):
    plugin.interactions_set_engine("ds")

    with pytest.raises(ValueError):
        plugin.interactions_set_cutoff("hbond_angle", value)


@pytest.mark.parametrize("name", ["all", "everything", "interactions or all", ""])
def test_destructive_group_names_are_rejected(name):
    with pytest.raises(ValueError):
        plugin._validate_group_name(name)


def test_csv_writer_neutralizes_spreadsheet_formulas(tmp_path):
    path = tmp_path / "safe.csv"

    plugin._write_csv(path, ["label", "value"], [["=CMD()", -1.5]])

    content = path.read_text(encoding="utf-8-sig")
    assert "'=CMD()" in content
    assert "-1.5" in content


def test_occupancy_rejects_a_large_combined_frame_budget(monkeypatch):
    calls = []

    def compute(*_args):
        calls.append(1)
        plugin._last_compute_cost[0] = plugin.MAX_OCCUPANCY_COMPUTE_COST
        return [], False

    monkeypatch.setattr(plugin, "_compute_interactions", compute)
    monkeypatch.setattr(
        plugin.cmd,
        "count_states",
        lambda _selection: 2,
        raising=False,
    )
    monkeypatch.setattr(
        plugin.cmd,
        "set_color",
        lambda *_args, **_kwargs: None,
        raising=False,
    )

    with pytest.raises(ValueError, match="combined processing budget"):
        plugin.interactions_occupancy(
            "receptor",
            "ligand",
            start=1,
            end=2,
        )

    assert len(calls) == 1


def test_occupancy_accumulates_variable_cost_across_frames(monkeypatch):
    costs = iter((1, plugin.MAX_OCCUPANCY_COMPUTE_COST))

    def compute(*_args):
        plugin._last_compute_cost[0] = next(costs)
        return [], False

    monkeypatch.setattr(plugin, "_compute_interactions", compute)
    monkeypatch.setattr(
        plugin.cmd,
        "count_states",
        lambda _selection: 2,
        raising=False,
    )
    monkeypatch.setattr(
        plugin.cmd,
        "set_color",
        lambda *_args, **_kwargs: None,
        raising=False,
    )

    with pytest.raises(ValueError, match="combined processing budget"):
        plugin.interactions_occupancy(
            "receptor",
            "ligand",
            start=1,
            end=2,
        )

    plugin.interactions_set_cutoff("hbond_dist", 3.9)
    assert plugin.interactions_parity_status()["active"] is False

    plugin.interactions_set_cutoff("reset", 0)
    assert plugin.interactions_parity_status()["active"] is True


def test_pi_sigma_accepts_observed_hydrogen_centroid_distance():
    plugin.interactions_set_engine("ds")
    carbon = _atom(1, "C", (0, 0, 3.85), "C7")
    hydrogen = _atom(2, "H", (0, 0, 2.76), "H16")
    carbon.neighbors = [hydrogen]
    feature = {"rings": [], "sigma_donors": [(carbon, [hydrogen])]}
    result = plugin.detect_pi_sigma({"rings": [_ring()], "sigma_donors": []}, feature)
    assert len(result) == 1
    assert result[0]["hydrogen_centroid_distance_A"] == 2.76


def test_pi_donor_accepts_observed_theta_limit():
    plugin.interactions_set_engine("ds")
    donor = _atom(3, "N", (3.41, 0, 3.54), "N8")
    hydrogen = _atom(4, "H", (2.7, 0, 2.81), "H18")
    donor.neighbors = [hydrogen]
    feature = {"rings": [], "donors": [(donor, [hydrogen])]}
    result = plugin.detect_pi_donor_hbond({"rings": [_ring()], "donors": []}, feature)
    assert len(result) == 1
    assert result[0]["theta_deg"] < 45.0


def test_ds_like_filter_rejects_long_saltbridge():
    interaction = {
        "type": "saltbridge",
        "subtype": "",
        "a_point": np.asarray((0.0, 0.0, 0.0)),
        "b_point": np.asarray((4.3, 0.0, 0.0)),
        "dist": 4.3,
    }

    assert plugin._matches_ds_like(interaction) is False


def test_ds_like_filter_keeps_short_saltbridge():
    interaction = {
        "type": "saltbridge",
        "subtype": "",
        "a_point": np.asarray((0.0, 0.0, 0.0)),
        "b_point": np.asarray((3.8, 0.0, 0.0)),
        "dist": 3.8,
    }

    assert plugin._matches_ds_like(interaction) is True


def test_ds_like_filter_matches_the_bundled_docklens_profile_logic():
    for interaction_type in plugin.VALID_TYPES:
        interaction = {
            "type": interaction_type,
            "subtype": "",
            "a_point": np.asarray((0.0, 0.0, 0.0)),
            "b_point": np.asarray((3.5, 0.0, 0.0)),
            "dist": 3.5,
        }
        detail = types.SimpleNamespace(
            interaction_type=interaction_type,
            distance_A=interaction["dist"],
        )
        expected = plugin._docklens_analysis_profiles.detail_matches_profile(
            detail, "ds_like"
        )

        assert plugin._matches_ds_like(interaction) is expected


def test_ds_inferred_hbond_remains_available_without_explicit_hydrogen():
    plugin.interactions_set_engine("ds")
    donor = _atom(1, "O", (0, 0, 0), "OG", sybyl_type="O.3", resn="SER")
    donor_carbon = _atom(2, "C", (-1.3, 0, 0), "CB", sybyl_type="C.3", resn="SER")
    acceptor = _atom(3, "O", (3.0, 0, 0), "O1", sybyl_type="O.2")
    acceptor_base = _atom(4, "C", (4.2, 0, 0), "C1", sybyl_type="C.2")
    _bond(donor, donor_carbon)
    _bond(acceptor, acceptor_base, "2")

    donor_features = plugin.classify(
        [donor, donor_carbon], [], has_h=False, chemistry_profile="dsv"
    )
    acceptor_features = plugin.classify(
        [acceptor, acceptor_base], [], has_h=False, chemistry_profile="dsv"
    )

    records = plugin.detect_hbond(donor_features, acceptor_features, False)

    assert len(records) == 1
    assert records[0]["chemistry_basis"] == "inferred_hydrogen"
    assert records[0]["confidence"] == "medium"
