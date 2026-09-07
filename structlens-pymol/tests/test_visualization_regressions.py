from unittest.mock import Mock

import pytest

from structlens_pymol_plugin.visualization import PyMOLVisualization


def reader():
    result = Mock()
    result.manifest = {'analysis_id': 'abc'}
    result.reference_id = 'ref'
    result.target_ids = ('one', 'two')
    result.structure_entry.side_effect = lambda identifier: f'{identifier}.cif'
    result.structure_bytes.return_value = b'data_test'
    result.transforms.return_value = {}
    return result


def residue(number=7, chain='A'):
    return {'chain_id': chain, 'auth_seq_id': number, 'insertion_code': ''}


def test_cif_uses_cif_parser():
    cmd = Mock()
    visual = PyMOLVisualization(cmd, reader())
    visual.load_structures()
    assert cmd.read_cifstr.call_count == 3
    cmd.read_pdbstr.assert_not_called()


def test_transform_matches_row_vector_coordinates_and_leaves_reference_fixed():
    source = reader()
    source.transforms.return_value = {'target': {'rotation': [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], 'translation': [4, 5, 6]}}
    source.target_ids = ('one',)
    cmd = Mock()
    visual = PyMOLVisualization(cmd, source)
    visual.apply_transforms(['ref_object', 'target_object'])
    assert cmd.transform_object.call_count == 1
    name, matrix = cmd.transform_object.call_args.args
    assert name == 'target_object'
    xyz = (2, 3, 4)
    transformed = [sum(matrix[4*i+j] * xyz[j] for j in range(3)) + matrix[4*i+3] for i in range(3)]
    assert transformed == pytest.approx([1, 7, 10])


def test_semantic_selection_filters_target_and_scopes_deletions_to_reference():
    source = reader()
    source.correspondence.return_value = [
        {'target_id': 'one', 'status': 'substitution', 'target': residue(7)},
        {'target_id': 'two', 'status': 'substitution', 'target': residue(99)},
        {'target_id': 'one', 'status': 'deletion', 'reference': residue(8), 'target': None},
    ]
    cmd = Mock()
    visual = PyMOLVisualization(cmd, source)
    names = visual.create_semantic_selections()
    selected = dict(call.args for call in cmd.select.call_args_list)
    assert 'SL_abc_T001' in selected[names['mutations']]
    assert '99' not in selected[names['mutations']]
    assert 'SL_abc_REF' in selected[names['deletions']]
    assert 'resi 8' in selected[names['deletions']]
    assert visual.focus_selection() == names['mutations']
    visual.set_active_target('two')
    names_two = visual.create_semantic_selections()
    assert names_two['mutations'] != names['mutations']


def test_blank_chain_is_selectable_without_selecting_other_objects():
    source = reader()
    source.correspondence.return_value = [{'target_id': 'one', 'status': 'substitution', 'target': residue(chain='')}]
    cmd = Mock()
    names = PyMOLVisualization(cmd, source).create_semantic_selections()
    expression = dict(call.args for call in cmd.select.call_args_list)[names['mutations']]
    assert 'chain ""' in expression
    assert 'SL_abc_T001' in expression


def test_vectors_use_valid_cgo_cylinders_and_active_target():
    source = reader()
    source.vectors.return_value = {'vectors': [
        {'target_id': target, 'start_xyz': [0, 0, 0], 'end_xyz': [1, 2, 3], 'magnitude_angstrom': 4}
        for target in ('one', 'two')
    ]}
    cmd = Mock()
    PyMOLVisualization(cmd, source).draw_displacement_vectors()
    primitive = cmd.load_cgo.call_args.args[0]
    assert primitive[0] == 9.0
    assert len(primitive) == 14
    assert primitive[1:7] == [0, 0, 0, 1, 2, 3]


def test_interactions_and_sites_scope_to_active_objects():
    source = reader()
    source.interactions.return_value = {'differences': [
        {'target_id': 'one', 'change': 'lost', 'reference_record': {'residue_a': residue(8)}},
        {'target_id': 'two', 'change': 'lost', 'reference_record': {'residue_a': residue(99)}},
    ]}
    source.sites.return_value = {'site': {'reference': [residue(1)], 'target': [residue(2)]}}
    cmd = Mock()
    visual = PyMOLVisualization(cmd, source)
    names = visual.create_interaction_selections()
    selected = dict(call.args for call in cmd.select.call_args_list)
    assert 'SL_abc_REF' in selected[names['lost']]
    assert '99' not in selected[names['lost']]
    names = visual.create_site_selections('site')
    selected = dict(call.args for call in cmd.select.call_args_list)
    assert 'SL_abc_REF' in selected[names['reference']]
    assert 'SL_abc_T001' in selected[names['target']]


def test_runtime_namespace_preserves_existing_objects_and_reset_only_owns_new_names():
    source = reader()
    cmd = Mock()
    cmd.get_names.return_value = ['SL_abc_REF', 'SL_abc_T001', 'SL_abc_1_REF']
    visual = PyMOLVisualization(cmd, source)
    names = visual.load_structures()
    assert visual.analysis_id == 'abc_2'
    assert not set(names) & set(cmd.get_names.return_value)
    visual.reset()
    assert [call.args[0] for call in cmd.delete.call_args_list] == list(names)


def test_pdb_loader_and_materialized_cif_fallback_cleanup(tmp_path):
    source = reader()
    source.structure_entry.side_effect = lambda identifier: f'{identifier}.pdb'
    cmd = Mock()
    PyMOLVisualization(cmd, source).load_structures()
    assert cmd.read_pdbstr.call_count == 3
    source.structure_entry.side_effect = lambda identifier: f'{identifier}.cif'
    path = tmp_path / 'materialized.cif'
    path.write_bytes(b'data_test')
    source.materialize_structure.return_value = path
    cmd = Mock(spec=['load', 'transform_object'])
    PyMOLVisualization(cmd, source).load_structures()
    assert cmd.load.call_count == 3
    assert not path.exists()


def test_legacy_transform_is_not_applied_to_multiple_targets():
    source = reader()
    source.transforms.return_value = {'target': {'rotation': [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 'translation': [0, 0, 0]}}
    cmd = Mock()
    PyMOLVisualization(cmd, source).apply_transforms(['ref', 'first', 'second'])
    cmd.transform_object.assert_not_called()


def test_failed_load_is_owned_for_transaction_cleanup():
    source = reader()
    cmd = Mock()
    cmd.read_cifstr.side_effect = RuntimeError('partial load')
    visual = PyMOLVisualization(cmd, source)
    with pytest.raises(RuntimeError, match='partial load'):
        visual.load_structures()
    visual.reset()
    cmd.delete.assert_called_once_with('SL_abc_REF')


def test_switch_target_then_focus_uses_created_selection():
    source = reader()
    source.correspondence.return_value = [
        {'target_id': 'one', 'status': 'substitution', 'target': residue(7)},
        {'target_id': 'two', 'status': 'substitution', 'target': residue(99)},
    ]
    cmd = Mock()
    visual = PyMOLVisualization(cmd, source)
    visual.create_semantic_selections()
    visual.set_active_target('two')
    focused = visual.focus_selection()
    selected = dict(call.args for call in cmd.select.call_args_list)
    assert focused in selected
    assert 'SL_abc_T002' in selected[focused]
    assert 'resi 99' in selected[focused]
