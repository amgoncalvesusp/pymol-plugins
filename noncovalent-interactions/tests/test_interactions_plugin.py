"""Regression tests for the Discovery Studio-calibrated PyMOL detectors.

The plug-in normally runs inside PyMOL.  A tiny command stub lets its pure
geometry and feature code be tested in a regular Python environment.
"""

import importlib.util
import os
import sys
import types
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


def _atom(idx, elem, coord, name):
    atom = object.__new__(plugin.Atom)
    atom.idx = idx
    atom.elem = elem
    atom.name = name
    atom.resn = "LIG"
    atom.resi = "1"
    atom.chain = ""
    atom.segi = ""
    atom.coord = np.asarray(coord, dtype=float)
    atom.fcharge = 0
    atom.neighbors = []
    return atom


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
    ds = plugin.CUTOFF_PROFILES["ds"]
    assert ds["pialkyl_dist"] == 4.9
    assert ds["alkyl_dist"] == 4.2
    assert ds["metal_dist"] == 3.0
    assert ds["pi_sigma_h_centroid_dist"] == 4.3
    assert ds["pi_donor_h_centroid_dist"] == 4.1
    assert ds["pi_lone_pair_dist"] == 3.5
    assert ds["pi_lone_pair_angle"] == 30.0
    assert {"pi_sigma", "pi_donor_hbond", "pi_lone_pair"} <= set(plugin.VALID_TYPES)


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
