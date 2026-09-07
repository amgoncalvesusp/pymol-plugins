"""Regression tests for the Discovery Studio-calibrated PyMOL detectors.

The plug-in normally runs inside PyMOL.  A tiny command stub lets its pure
geometry and feature code be tested in a regular Python environment.
"""

import importlib.util
import importlib
import hashlib
import inspect
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


_CANONICAL_PROFILES = ("plip", "luna", "dsv", "luna_dsv")


def _canonical_profile_name(name):
    value = str(name).strip().lower()
    return "dsv" if value == "ds" else value


def _profile_snapshot(profiles, canonical_name):
    if canonical_name in profiles:
        return profiles[canonical_name]
    if canonical_name == "dsv" and "ds" in profiles:
        return profiles["ds"]
    pytest.fail("missing cutoff profile %r" % canonical_name)


_EXPECTED_CANONICAL_INTERACTION_COLORS = {
    "hbond": ("skyblue", "Hydrogen bond (conventional)"),
    "carbon_hbond": ("bluishgreen", "Carbon H-bond (weak, C-H...O/N)"),
    "saltbridge": ("vermillion", "Salt bridge / ionic"),
    "attractive_charge": ("orange", "Attractive charge interaction"),
    "charge_repulsion": ("vermillion", "Repulsive charge interaction"),
    "pipi": ("reddishpurple", "pi-pi stacking (sandwich/T-shaped)"),
    "pication": ("yellow", "pi-cation"),
    "pialkyl": ("orange", "pi-alkyl"),
    "pi_sigma": ("reddishpurple", "pi-sigma"),
    "alkyl": ("blue", "Alkyl-alkyl (hydrophobic)"),
    "halogen": ("black", "Halogen bond"),
    "metal": ("vermillion", "Metal coordination"),
    "water_bridge": ("skyblue", "Water-mediated H-bond"),
    "pi_sulfur": ("reddishpurple", "pi-sulfur"),
    "pi_anion": ("yellow", "pi-anion"),
    "pi_donor_hbond": ("bluishgreen", "pi-donor hydrogen bond"),
    "pi_lone_pair": ("skyblue", "Lone pair-pi"),
    "chalcogen": ("black", "Chalcogen bond (S/Se/Te)"),
}


_APPEARANCE_PARAMETER_ALIASES = {
    "protein_selection": (
        "protein_selection",
        "receptor_selection",
        "protein",
        "sel1",
    ),
    "ligand_selection": ("ligand_selection", "ligand", "sel2"),
    "protein_color": ("protein_color", "receptor_color"),
    "ligand_color": ("ligand_color",),
    "interacting_residue_color": (
        "interacting_residue_color",
        "interaction_residue_color",
        "residue_color",
    ),
    "stick_radius": ("stick_radius",),
    "nonbond_sphere_size": (
        "nonbond_sphere_scale",
        "nonbond_sphere_radius",
        "nonbonded_size",
        "sphere_scale",
        "sphere_radius",
    ),
    "dash_thickness": ("dash_thickness", "thickness"),
    "dash_scale": ("dash_scale",),
    "label_size": ("label_size",),
    "cartoon_transparency": ("cartoon_transparency",),
    "background_color": ("background_color", "bg_color"),
    "show_hydrogens": ("show_hydrogens", "hydrogens"),
    "transparency": (
        "transparency",
        "object_transparency",
        "global_transparency",
    ),
}

_APPEARANCE_DEFAULTS = {
    "protein_selection": "polymer and chain A",
    "ligand_selection": "organic and resn LIG",
    "protein_color": "gray70",
    "ligand_color": "orange",
    "interacting_residue_color": "marine",
    "stick_radius": 0.22,
    "nonbond_sphere_size": 0.31,
    "dash_thickness": 0.09,
    "dash_scale": 1.5,
    "label_size": 18,
    "cartoon_transparency": 0.25,
    "background_color": "white",
    "show_hydrogens": True,
    "transparency": 0.15,
}

_OPTIONAL_RENDER_PARAMETERS = {
    "ambient": 0.35,
    "specular": 0.2,
    "ray_shadows": 0,
    "ray_opaque_background": 0,
    "antialias": 2,
}


def _appearance_parameter_name(semantic):
    parameters = inspect.signature(plugin.interactions_set_appearance).parameters
    for candidate in _APPEARANCE_PARAMETER_ALIASES[semantic]:
        if candidate in parameters:
            return candidate
    pytest.fail(
        "appearance API does not expose %s (accepted names: %s)"
        % (semantic, ", ".join(_APPEARANCE_PARAMETER_ALIASES[semantic]))
    )


def _appearance_kwargs(**overrides):
    values = dict(_APPEARANCE_DEFAULTS)
    values.update(overrides)
    kwargs = {
        _appearance_parameter_name(semantic): value
        for semantic, value in values.items()
    }
    parameters = inspect.signature(plugin.interactions_set_appearance).parameters
    if "group_name" in parameters:
        kwargs["group_name"] = "interactions"
    for name, value in _OPTIONAL_RENDER_PARAMETERS.items():
        if name in parameters:
            kwargs[name] = value
    return kwargs


def _source_bytes(path_or_data):
    """Content with line endings normalised.

    Git rewrites LF to CRLF in a Windows checkout while the installable
    zip ships LF, so raw-byte equality asserts the checkout style rather
    than the reviewed content.
    """
    data = (
        path_or_data
        if isinstance(path_or_data, bytes)
        else path_or_data.read_bytes()
    )
    return data.replace(b"\r\n", b"\n")


def _record_pymol_mutations(monkeypatch):
    calls = []

    def recorder(command):
        def record(*args, **kwargs):
            calls.append((command, args, kwargs))

        return record

    for command in ("bg_color", "color", "hide", "set", "show"):
        monkeypatch.setattr(plugin.cmd, command, recorder(command), raising=False)
    return calls


def _setting_was_applied(calls, names, expected):
    for command, args, _kwargs in calls:
        if command != "set" or len(args) < 2 or args[0] not in names:
            continue
        try:
            if float(args[1]) == pytest.approx(float(expected)):
                return True
        except (TypeError, ValueError):
            continue
    return False


def _color_was_applied(calls, color, selection_fragment):
    for command, args, _kwargs in calls:
        if command == "color" and len(args) >= 2:
            if args[0] == color and selection_fragment in str(args[1]).lower():
                return True
        if command == "set" and len(args) >= 3:
            if (
                str(args[0]).endswith("_color")
                and args[1] == color
                and selection_fragment in str(args[2]).lower()
            ):
                return True
    return False


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


def test_gui_source_exposes_four_profiles_and_full_appearance_controls():
    source = inspect.getsource(plugin.run_plugin_gui).lower()

    for profile in _CANONICAL_PROFILES:
        assert profile in source

    required_control_terms = (
        ("protein", "color"),
        ("ligand", "color"),
        ("residue", "color"),
        ("stick", "radius"),
        ("nonbond", "sphere"),
        ("dash", "thickness"),
        ("dash", "scale"),
        ("label", "size"),
        ("cartoon", "transparency"),
        ("background", "color"),
        ("hydrogen",),
    )
    for terms in required_control_terms:
        assert all(term in source for term in terms), terms
    assert "transparency" in source.replace("cartoon_transparency", "")


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


def test_engine_choices_expose_the_four_canonical_profiles():
    choices = {
        _canonical_profile_name(name) for name in plugin.DETECTION_ENGINES
    }

    assert choices == set(_CANONICAL_PROFILES)


@pytest.mark.parametrize("profile", _CANONICAL_PROFILES)
def test_each_canonical_profile_can_be_selected(profile):
    original = plugin._active_engine[0]
    try:
        plugin.interactions_set_engine("plip")
        plugin.interactions_set_engine(profile)

        assert _canonical_profile_name(plugin._active_engine[0]) == profile
        assert plugin.CUTOFFS == dict(
            _profile_snapshot(plugin.CUTOFF_PROFILES, profile)
        )
    finally:
        plugin.interactions_set_engine(original)


def test_legacy_ds_alias_selects_the_dsv_profile():
    original = plugin._active_engine[0]
    try:
        plugin.interactions_set_engine("plip")
        plugin.interactions_set_engine("ds")

        assert _canonical_profile_name(plugin._active_engine[0]) == "dsv"
        assert plugin.CUTOFFS == dict(
            _profile_snapshot(plugin.CUTOFF_PROFILES, "dsv")
        )
    finally:
        plugin.interactions_set_engine(original)


@pytest.mark.parametrize("profile", _CANONICAL_PROFILES)
def test_cutoff_profiles_match_bundled_docklens_snapshots(profile):
    core_profiles = plugin._docklens_core.HBOND_PRESETS
    core_choices = {_canonical_profile_name(name) for name in core_profiles}

    assert profile in core_choices
    expected = dict(plugin._docklens_core.cutoffs_for_preset(profile))
    actual = dict(_profile_snapshot(plugin.CUTOFF_PROFILES, profile))
    assert actual == expected


def test_canonical_interaction_colors_exactly_match_bundled_core():
    assert (
        plugin._docklens_core.INTERACTION_COLORS
        == _EXPECTED_CANONICAL_INTERACTION_COLORS
    )
    assert plugin.INTERACTION_COLORS == plugin._docklens_core.INTERACTION_COLORS
    assert plugin.VALID_TYPES == list(_EXPECTED_CANONICAL_INTERACTION_COLORS)


def test_luna_charge_families_do_not_double_count_one_ionic_pair():
    cation = _atom(1, "N", (0, 0, 0), "N1", fcharge=1)
    anion = _atom(2, "O", (3.0, 0, 0), "O1", fcharge=-1)
    cation.side = "receptor"
    anion.side = "ligand"

    records = plugin._docklens_core.compute_interactions(
        [cation],
        [anion],
        types=["saltbridge", "attractive_charge"],
        chemistry_profile="luna",
    )

    assert [record["type"] for record in records] == ["saltbridge"]


def test_appearance_api_applies_scene_and_interaction_settings(
    monkeypatch,
):
    calls = _record_pymol_mutations(monkeypatch)
    monkeypatch.setattr(
        plugin,
        "_dash_base",
        {"hbond_dash": (0.35, 0.35)},
    )

    plugin.interactions_set_appearance(**_appearance_kwargs())

    assert _color_was_applied(calls, "gray70", "polymer")
    assert _color_was_applied(calls, "orange", "organic")
    assert _color_was_applied(calls, "marine", "residu")
    assert _setting_was_applied(calls, {"stick_radius"}, 0.22)
    assert _setting_was_applied(
        calls,
        {
            "nonbond_sphere_radius",
            "nonbond_sphere_scale",
            "nonbonded_size",
            "sphere_radius",
            "sphere_scale",
        },
        0.31,
    )
    assert _setting_was_applied(calls, {"dash_radius"}, 0.09)
    assert _setting_was_applied(calls, {"dash_length"}, 0.525)
    assert _setting_was_applied(calls, {"dash_gap"}, 0.525)
    assert _setting_was_applied(calls, {"label_size"}, 18)
    assert _setting_was_applied(calls, {"cartoon_transparency"}, 0.25)
    assert _setting_was_applied(calls, {"transparency"}, 0.15)
    assert any(
        command == "bg_color" and args and args[0] == "white"
        for command, args, _kwargs in calls
    )
    assert any(
        command == "show"
        and args
        and any(
            marker in " ".join(str(arg).lower() for arg in args)
            for marker in ("elem h", "hydro")
        )
        for command, args, _kwargs in calls
    )

    parameters = inspect.signature(
        plugin.interactions_set_appearance
    ).parameters
    for name, expected in _OPTIONAL_RENDER_PARAMETERS.items():
        if name in parameters:
            assert _setting_was_applied(calls, {name}, expected)


def test_appearance_api_can_hide_hydrogens(monkeypatch):
    calls = _record_pymol_mutations(monkeypatch)
    monkeypatch.setattr(plugin, "_dash_base", {})

    plugin.interactions_set_appearance(
        **_appearance_kwargs(show_hydrogens=False)
    )

    assert any(
        command == "hide"
        and args
        and any(
            marker in " ".join(str(arg).lower() for arg in args)
            for marker in ("elem h", "hydro")
        )
        for command, args, _kwargs in calls
    )


@pytest.mark.parametrize(
    ("semantic", "invalid"),
    (
        ("protein_selection", "polymer; delete all"),
        ("ligand_selection", "organic\nhide everything"),
        ("protein_color", "red; delete all"),
        ("ligand_color", "#12GG00"),
        ("interacting_residue_color", ""),
    ),
)
def test_appearance_rejects_invalid_text_without_pymol_commands(
    monkeypatch,
    semantic,
    invalid,
):
    calls = _record_pymol_mutations(monkeypatch)

    with pytest.raises(ValueError):
        plugin.interactions_set_appearance(
            **_appearance_kwargs(**{semantic: invalid})
        )

    assert calls == []


@pytest.mark.parametrize(
    ("semantic", "invalid"),
    (
        ("stick_radius", 0.0),
        ("nonbond_sphere_size", float("nan")),
        ("dash_thickness", -0.01),
        ("dash_scale", float("inf")),
        ("label_size", 0),
        ("cartoon_transparency", 1.01),
        ("transparency", -0.01),
    ),
)
def test_appearance_rejects_invalid_numbers_without_pymol_commands(
    monkeypatch,
    semantic,
    invalid,
):
    calls = _record_pymol_mutations(monkeypatch)

    with pytest.raises(ValueError):
        plugin.interactions_set_appearance(
            **_appearance_kwargs(**{semantic: invalid})
        )

    assert calls == []


def test_ds_profile_matches_calibrated_geometry():
    plip = plugin.CUTOFF_PROFILES["plip"]
    dsv = _profile_snapshot(plugin.CUTOFF_PROFILES, "dsv")
    assert plugin.DSV_PARITY_CONTRACT == "docklens-scientific-profiles-2026.08"
    assert plip == dict(plugin._docklens_core.cutoffs_for_preset("plip"))
    assert dsv == dict(plugin._docklens_core.cutoffs_for_preset("dsv"))
    assert plugin.VALID_TYPES == plugin._docklens_core.VALID_TYPES
    assert plugin.INTERACTION_COLORS == plugin._docklens_core.INTERACTION_COLORS
    assert dsv["hbond_dist"] == 3.4
    assert dsv["hbond_acceptor_angle"] == 90.0
    assert dsv["carbon_hbond_dist"] == 3.8
    assert dsv["pialkyl_dist"] == 5.5
    assert dsv["alkyl_dist"] == 5.5
    assert dsv["metal_dist"] == 3.0
    assert dsv["pi_sigma_carbon_dist"] == 4.0
    assert dsv["pi_donor_dist"] == 4.2
    assert dsv["pi_lone_pair_dist"] == 3.0
    assert dsv["pi_lone_pair_angle"] == 45.0
    assert {"pi_sigma", "pi_donor_hbond", "pi_lone_pair"} <= set(plugin.VALID_TYPES)


def test_bundled_core_matches_the_plugin_reviewed_digest():
    path = (
        Path(__file__).parents[1]
        / "pymol_interactions_plugin"
        / "docklens_core.py"
    )
    digest = hashlib.sha256(_source_bytes(path)).hexdigest()

    assert digest == plugin._EXPECTED_DOCKLENS_CORE_SHA256


def test_bundled_analysis_profile_matches_the_plugin_reviewed_digest():
    path = (
        Path(__file__).parents[1]
        / "pymol_interactions_plugin"
        / "docklens_analysis_profiles.py"
    )
    digest = hashlib.sha256(_source_bytes(path)).hexdigest()

    assert digest == plugin._EXPECTED_ANALYSIS_PROFILE_SHA256


def test_unreviewed_fallback_module_is_rejected_before_import(tmp_path):
    path = tmp_path / "docklens_core.py"
    path.write_text("raise RuntimeError('must never execute')\n", encoding="utf-8")

    with pytest.raises(ImportError, match="integrity"):
        plugin._verify_reviewed_source(path, "0" * 64)


def test_compute_budget_tracks_linear_water_screening():
    # Each water is screened once against each side, not once per
    # receptor x ligand pair. The old cartesian model rejected ordinary
    # solvated pockets, so detection returned nothing at all.
    assert plugin._estimate_compute_cost(10, 20, 3, include_water=True) == 290
    assert plugin._estimate_compute_cost(10, 20, 3, include_water=False) == 200


def test_solvated_pocket_stays_within_the_per_frame_budget():
    cost = plugin._estimate_compute_cost(4000, 40, 150, include_water=True)

    assert cost <= plugin.MAX_COMPUTE_COST_PER_FRAME


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
    assert packaged.CUTOFF_PROFILES["plip"] == dict(
        packaged._docklens_core.cutoffs_for_preset("plip")
    )
    assert _profile_snapshot(packaged.CUTOFF_PROFILES, "dsv") == dict(
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
            assert _source_bytes(
                archive.read("pymol_interactions_plugin/" + name)
            ) == _source_bytes(root / "pymol_interactions_plugin" / name)


def test_plugin_copies_and_zip_bundle_are_synchronized():
    root = Path(__file__).parents[1]
    package = root / "pymol_interactions_plugin"
    root_plugin = _source_bytes(root / "interactions_plugin.py")
    packaged_plugin = _source_bytes(package / "interactions_plugin.py")

    assert root_plugin == packaged_plugin
    with zipfile.ZipFile(root / "pymol_interactions_plugin.zip") as archive:
        assert _source_bytes(
            archive.read("pymol_interactions_plugin/interactions_plugin.py")
        ) == root_plugin
        for module_name in (
            "docklens_core.py",
            "docklens_analysis_profiles.py",
        ):
            assert _source_bytes(
                archive.read("pymol_interactions_plugin/" + module_name)
            ) == _source_bytes(package / module_name)


def test_dsv_is_the_default_engine_for_docklens_figure_parity():
    assert _canonical_profile_name(plugin._active_engine[0]) == "dsv"


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
    acceptor = _atom(3, "O", (3.3, 0, 0), "O", sybyl_type="O.2")
    acceptor_base = _atom(4, "C", (3.3, 1, 0), "C", sybyl_type="C.2")
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
    assert record["hydrogen_acceptor_distance_A"] == pytest.approx(2.3)
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
    acceptor = _atom(4, "O", (3.3, 0, 0), "O", sybyl_type="O.2")
    acceptor_base = _atom(5, "C", (3.3, 0, 1), "C", sybyl_type="C.2")
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
    theta = np.deg2rad(44.0)
    direction = (np.sin(theta), 0.0, np.cos(theta))
    donor = _atom(3, "N", tuple(4.1 * value for value in direction), "N8")
    hydrogen = _atom(4, "H", tuple(3.1 * value for value in direction), "H18")
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


# --------------------------------------------------------------------------
# 0.7.0 regressions: the plugin had stopped reporting interactions entirely
# --------------------------------------------------------------------------


def test_reviewed_digest_accepts_a_windows_checkout(tmp_path):
    """A CRLF checkout must still load: LF and CRLF are the same source."""
    reviewed = b"value = 1\norther = 2\n"
    digest = hashlib.sha256(reviewed).hexdigest()
    windows_copy = tmp_path / "docklens_core.py"
    windows_copy.write_bytes(reviewed.replace(b"\n", b"\r\n"))

    assert plugin._verify_reviewed_source(windows_copy, digest) == windows_copy


def test_reviewed_digest_still_rejects_modified_content(tmp_path):
    tampered = tmp_path / "docklens_core.py"
    tampered.write_bytes(b"raise RuntimeError('must never execute')")

    with pytest.raises(ImportError, match="integrity"):
        plugin._verify_reviewed_source(tampered, "0" * 64)


def _stub_scene_commands(monkeypatch):
    """No-op the PyMOL scene calls detect_interactions makes."""
    for command in (
        "set_color",
        "delete",
        "disable",
        "enable",
        "group",
        "hide",
        "show",
        "select",
        "deselect",
        "set",
        "color",
        "distance",
        "pseudoatom",
    ):
        monkeypatch.setattr(
            plugin.cmd, command, lambda *_a, **_k: None, raising=False
        )
    monkeypatch.setattr(
        plugin.cmd, "get_names", lambda *_a, **_k: [], raising=False
    )
    monkeypatch.setattr(
        plugin.cmd, "count_atoms", lambda *_a, **_k: 0, raising=False
    )


def test_busy_pocket_keeps_drawing_instead_of_raising(monkeypatch):
    """A crowded site must still show interactions, closest ones first."""
    _stub_scene_commands(monkeypatch)
    crowded = [
        {
            "type": "alkyl",
            "subtype": "",
            "a_label": "LEU%d_CD1" % index,
            "b_label": "LIG1_C1",
            "a_point": np.array([float(index), 0.0, 0.0]),
            "b_point": np.array([float(index), 1.0, 0.0]),
            "a_sele": "(resn LEU)",
            "b_sele": "(resn LIG)",
            "dist": float(plugin.MAX_DRAWN_INTERACTIONS + 10 - index),
        }
        for index in range(plugin.MAX_DRAWN_INTERACTIONS + 25)
    ]
    monkeypatch.setattr(
        plugin,
        "_compute_interactions",
        lambda *_args, **_kwargs: (crowded, True),
    )
    drawn = []
    monkeypatch.setattr(
        plugin,
        "_draw",
        lambda record, *_args, **_kwargs: drawn.append(record),
    )

    counts = plugin.detect_interactions()

    assert sum(counts.values()) == plugin.MAX_DRAWN_INTERACTIONS
    assert len(drawn) == plugin.MAX_DRAWN_INTERACTIONS
    assert max(record["dist"] for record in drawn) < max(
        record["dist"] for record in crowded
    )


def test_appearance_survives_a_missing_residue_selection(monkeypatch):
    """Styling <group>_residues before any detection must not abort."""
    calls = _record_pymol_mutations(monkeypatch)

    def _color(color, selection, *_args, **_kwargs):
        if "residues" in str(selection):
            raise RuntimeError("Invalid selection name")
        calls.append(("color", (color, selection), {}))

    monkeypatch.setattr(plugin.cmd, "color", _color, raising=False)
    monkeypatch.setattr(plugin, "_dash_base", {"hbond_dash": (0.35, 0.35)})

    plugin.interactions_set_appearance(**_appearance_kwargs())

    assert _color_was_applied(calls, "gray70", "polymer")
    assert _color_was_applied(calls, "orange", "organic")
    assert _setting_was_applied(calls, {"dash_radius"}, 0.09)


# --------------------------------------------------------------------------
# 0.7.0 features: DockLens analysis views and the best viewing angle
# --------------------------------------------------------------------------


def test_analysis_views_match_the_bundled_docklens_names():
    assert set(plugin.ANALYSIS_PROFILES) == set(
        plugin._docklens_analysis_profiles.VALID_ANALYSIS_PROFILES
    )


def test_analysis_view_filters_records_like_docklens(monkeypatch):
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
                        (4.6, 0, 0),
                        "O1",
                        sybyl_type="O.co2",
                        fcharge=-1,
                    )
                ],
                False,
            )
        ),
    )
    plugin.interactions_set_engine("plip")
    try:
        plugin.interactions_set_analysis_profile("complete")
        complete, _has_h = plugin._compute_interactions(
            "receptor", "ligand", ["saltbridge"], 1
        )
        plugin.interactions_set_analysis_profile("ds_like")
        ds_like, _has_h = plugin._compute_interactions(
            "receptor", "ligand", ["saltbridge"], 1
        )
    finally:
        plugin.interactions_set_analysis_profile("complete")
        plugin.interactions_set_engine("dsv")

    assert len(complete) == 1
    # 4.6 A exceeds DockLens' DS reporting ceiling for salt bridges
    assert ds_like == []


def test_unknown_analysis_view_is_rejected_without_changing_state():
    try:
        assert plugin.interactions_set_analysis_profile("nonsense") is None
        assert plugin._active_analysis_profile[0] == "complete"
    finally:
        plugin.interactions_set_analysis_profile("complete")


def test_best_angle_looks_down_the_thin_axis_of_the_network(monkeypatch):
    """Interactions spread in x/y, thin in z -> camera looks along z."""
    points = [
        (np.array([0.0, 0.0, 0.0]), np.array([4.0, 0.0, 0.02])),
        (np.array([0.0, 3.0, 0.0]), np.array([4.0, 3.0, -0.02])),
        (np.array([2.0, 1.0, 0.01]), np.array([2.0, -3.0, 0.0])),
    ]
    monkeypatch.setattr(plugin, "_last_interaction_points", points)
    monkeypatch.setattr(
        plugin,
        "_last_detection_context",
        {"sel1": "polymer", "sel2": "organic", "group_name": "interactions"},
    )
    extents = {
        "polymer": ((0.0, 0.0, -6.0), (4.0, 3.0, -4.0)),
        "organic": ((0.0, 0.0, 4.0), (4.0, 3.0, 6.0)),
    }
    monkeypatch.setattr(
        plugin.cmd,
        "get_extent",
        lambda selection, *_a, **_k: extents.get(
            selection, ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        ),
        raising=False,
    )
    monkeypatch.setattr(
        plugin.cmd,
        "get_view",
        lambda *_a, **_k: tuple([1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0] + [0.0] * 9),
        raising=False,
    )
    applied = {}
    monkeypatch.setattr(
        plugin.cmd,
        "set_view",
        lambda view: applied.setdefault("view", list(view)),
        raising=False,
    )
    monkeypatch.setattr(
        plugin.cmd,
        "zoom",
        lambda *args, **_k: applied.setdefault("zoom", args),
        raising=False,
    )
    monkeypatch.setattr(
        plugin.cmd, "count_atoms", lambda *_a, **_k: 0, raising=False
    )

    result = plugin.interactions_best_angle()

    assert result is not None
    view_axis = np.asarray(result["view_axis"], dtype=float)
    # the thin direction of this network is z
    assert abs(float(view_axis[2])) > 0.99
    # the ligand (z > 0) must end up on the camera side of the receptor
    assert float(view_axis[2]) > 0.0
    assert result["planarity"] > 0.99
    assert applied["view"][0:9] == pytest.approx(
        [
            float(value)
            for row in (
                result["right"],
                result["up"],
                result["view_axis"],
            )
            for value in row
        ]
    )


def test_best_angle_can_report_without_moving_the_camera(monkeypatch):
    points = [
        (np.array([0.0, 0.0, 0.0]), np.array([4.0, 0.0, 0.0])),
        (np.array([0.0, 3.0, 0.0]), np.array([4.0, 3.0, 0.0])),
    ]
    monkeypatch.setattr(plugin, "_last_interaction_points", points)
    monkeypatch.setattr(
        plugin,
        "_last_detection_context",
        {"sel1": "polymer", "sel2": "organic", "group_name": "interactions"},
    )
    monkeypatch.setattr(
        plugin.cmd,
        "get_extent",
        lambda *_a, **_k: ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        raising=False,
    )
    moved = []
    monkeypatch.setattr(
        plugin.cmd, "set_view", lambda *_a, **_k: moved.append(True),
        raising=False,
    )

    result = plugin.interactions_best_angle(apply=0)

    assert result is not None
    assert moved == []


def test_best_angle_needs_a_detection_run_first(monkeypatch):
    monkeypatch.setattr(plugin, "_last_interaction_points", [])

    assert plugin.interactions_best_angle() is None


def test_gui_exposes_the_analysis_view_and_best_angle_controls():
    source = inspect.getsource(plugin.run_plugin_gui).lower()

    assert "analysis_profiles" in source
    assert "interactions_set_analysis_profile" in source
    assert "analysis view" in source
    assert "best angle" in source
    assert "interactions_best_angle" in source
    # detection must not sit behind a successful appearance pass
    detect_body = source.split("def _run():", 1)[1].split("def _run_occ", 1)[0]
    assert "if not _apply_appearance():" not in detect_body


def test_detection_records_geometry_for_the_best_angle(monkeypatch):
    _stub_scene_commands(monkeypatch)
    records = [
        {
            "type": "hbond",
            "subtype": "",
            "a_label": "SER90_OG",
            "b_label": "LIG1_O1",
            "a_point": np.array([0.0, 0.0, 0.0]),
            "b_point": np.array([2.8, 0.0, 0.0]),
            "a_sele": "(resn SER)",
            "b_sele": "(resn LIG)",
            "dist": 2.8,
        }
    ]
    monkeypatch.setattr(
        plugin,
        "_compute_interactions",
        lambda *_args, **_kwargs: (records, True),
    )
    monkeypatch.setattr(plugin, "_draw", lambda *_args, **_kwargs: None)

    plugin.detect_interactions("polymer", "organic")

    assert len(plugin._last_interaction_points) == 1
    assert plugin._last_detection_context["sel2"] == "organic"



@pytest.mark.parametrize(
    "threshold, expected",
    [(75, ["persistent"]), (50, ["persistent", "transient"]),
     (101, None), (float("nan"), None), (-1, None)],
)
def test_occupancy_draws_only_qualifying_final_frame_contacts(monkeypatch, threshold, expected):
    _stub_scene_commands(monkeypatch)
    monkeypatch.setattr(plugin.cmd, "count_states", lambda *_a: 2, raising=False)
    def record(label):
        return {"type": "hbond", "subtype": "", "a_label": label,
                "b_label": "LIG1_O1", "a_point": np.array([0., 0., 0.]),
                "b_point": np.array([2.8, 0., 0.]), "a_sele": "polymer",
                "b_sele": "organic", "dist": 2.8}
    frames = {1: [record("persistent"), record("absent_at_end")],
              2: [record("persistent"), record("transient")]}
    computed = []
    def compute(_s1, _s2, _types, state):
        computed.append(state)
        return frames[state], True
    monkeypatch.setattr(plugin, "_compute_interactions", compute)
    drawn = []
    monkeypatch.setattr(plugin, "_draw", lambda item, *_a: drawn.append(item["a_label"]))
    if expected is None:
        with pytest.raises(ValueError, match="threshold"):
            plugin.interactions_occupancy(threshold=threshold, draw=1)
        assert computed == []
    else:
        plugin.interactions_occupancy(threshold=threshold, draw=1)
        assert drawn == expected
        assert computed == [1, 2]


def test_plugin_manager_reports_current_release_version():
    package = importlib.import_module("pymol_interactions_plugin")
    assert package.__version__ == "0.7.1"


@pytest.mark.parametrize("target_type", ["object:molecule", "object:group"])
@pytest.mark.parametrize("operation", ["detect", "clear"])
def test_existing_user_object_is_not_deleted_by_group_name(monkeypatch, target_type, operation):
    _stub_scene_commands(monkeypatch)
    monkeypatch.setattr(plugin.cmd, "get_names", lambda *_a, **_k: ["protein"])
    monkeypatch.setattr(plugin.cmd, "get_type", lambda _name: target_type, raising=False)
    deleted = []
    monkeypatch.setattr(plugin.cmd, "delete", deleted.append)
    monkeypatch.setattr(plugin, "_compute_interactions", lambda *_a: ([], True))
    with pytest.raises(ValueError, match="already exists"):
        if operation == "detect":
            plugin.detect_interactions(group_name="protein")
        else:
            plugin.interactions_visibility("clear", group_name="protein")
    assert deleted == []
