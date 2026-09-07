import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from structlens_pymol_plugin import controller as module
from structlens_pymol_plugin import plugin
from structlens_pymol_plugin import __version__


def test_plugin_package_reports_current_release_version():
    assert __version__ == "0.3.1"


@pytest.mark.parametrize("stage", ["reader", "load", "selection"])
def test_failed_open_keeps_previous_analysis_and_cleans_candidate(monkeypatch, stage):
    controller = module.Controller(Mock())
    old_reader, old_view = Mock(), Mock()
    controller.reader, controller.visualization = old_reader, old_view
    reader, view = Mock(), Mock()
    reader_factory = Mock(return_value=reader)
    if stage == "reader":
        reader_factory.side_effect = ValueError("invalid bundle")
    elif stage == "load":
        view.load_structures.side_effect = ValueError("bad coordinates")
    else:
        view.create_semantic_selections.side_effect = ValueError("bad selection")
    monkeypatch.setattr(module, "BundleReader", reader_factory)
    monkeypatch.setattr(module, "PyMOLVisualization", Mock(return_value=view))
    with pytest.raises(ValueError):
        controller.open_bundle("new.structlens-pymol")
    assert controller.reader is old_reader
    assert controller.visualization is old_view
    old_reader.close.assert_not_called()
    old_view.reset.assert_not_called()
    if stage != "reader":
        reader.close.assert_called_once()
        view.reset.assert_called_once()


def test_successful_open_releases_previous_analysis(monkeypatch):
    controller = module.Controller(Mock())
    old_reader, old_view = Mock(), Mock()
    controller.reader, controller.visualization = old_reader, old_view
    reader, view = Mock(), Mock()
    monkeypatch.setattr(module, "BundleReader", Mock(return_value=reader))
    monkeypatch.setattr(module, "PyMOLVisualization", Mock(return_value=view))
    assert controller.open_bundle("new.structlens-pymol") is reader
    assert controller.visualization is view
    old_reader.close.assert_called_once()
    old_view.reset.assert_called_once()
    controller.reset()
    reader.close.assert_called_once()
    view.reset.assert_called_once()


def test_plugin_registers_menu_through_pymol_api(monkeypatch):
    command, add_menu = Mock(), Mock()
    monkeypatch.setitem(sys.modules, "pymol", SimpleNamespace(cmd=command))
    monkeypatch.setitem(sys.modules, "pymol.plugins", SimpleNamespace(addmenuitemqt=add_menu))
    monkeypatch.setattr(plugin, "_controller", None)
    plugin.__init_plugin__(object())
    command.extend.assert_called_once()
    add_menu.assert_called_once()
    assert add_menu.call_args.args[0] == "StructLens-PyMOL"
