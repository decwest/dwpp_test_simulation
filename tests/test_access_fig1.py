"""Validate the frozen filming conditions without ROS or robot access."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('fig1_experiment', ROOT/'scripts/dwvp_access_experiment.py')
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def test_fig1_frozen_trials_match_panels_and_share_starts(tmp_path):
    session = tmp_path/'fig1'
    manifest = experiment.prepare(
        session, ROOT/'params/hsrb_dwvp_access_params.yaml',
        config_file=ROOT/'params/dwvp_access_fig1.yaml', current_start=[2., 3., .4],
        conditions=['E1_lateral', 'E1_response_L0p33_T0p5', 'E1_orientation_quarter'],
        repeats=[1], ordered=True)
    assert manifest == json.loads((session/'manifest.json').read_text())
    assert manifest['ordered'] and not manifest.get('bidirectional')
    assert [(t['task'], t['controller']) for t in manifest['trials']] == [
        ('E1_lateral', 'VP_CLIP'), ('E1_response_L0p33_T0p5', 'DWVP'),
        ('E1_orientation_quarter', 'VP_CLIP'), ('E1_orientation_quarter', 'DWVP')]
    for index, trial in enumerate(manifest['trials']):
        experiment.load_trial(session, trial['id'])
        np.testing.assert_allclose(experiment.start_pose(manifest, trial), [2., 3., .4])
        params = yaml.safe_load((session/trial['params_file']).read_text())
        cs = params['controller_server']['ros__parameters']
        ctrl = cs[trial['controller']]
        smoother = params['velocity_smoother']['ros__parameters']
        expected_accel = [.22, .22, .6] if index < 2 else [.055, .055, .15]
        np.testing.assert_allclose([ctrl[f'max_accel_{axis}'] for axis in ('x', 'y', 'theta')], expected_accel)
        np.testing.assert_allclose(smoother['max_accel'], expected_accel)
        np.testing.assert_allclose(smoother['max_decel'], -np.array(expected_accel))
        assert ctrl['lateral_time_constant'] == (.5 if index == 1 else 0.)
        assert ctrl['use_dynamic_window_vector_pursuit'] == (trial['controller'] == 'DWVP')
        assert ctrl.get('use_uniform_velocity_scaling', False) is False
        assert ctrl['velocity_feedback'] == smoother['feedback'] == 'OPEN_LOOP'
        assert ctrl['vp_translation_speed'] == .22
        assert cs['controller_frequency'] == 30.
        if index < 2:
            assert ctrl['min_lookahead_dist'] == ctrl['max_lookahead_dist'] == .33
        else:
            assert (ctrl['lookahead_time'], ctrl['min_lookahead_dist'], ctrl['max_lookahead_dist']) == (.75, .11, .33)
    config = yaml.safe_load((session/'experiment_config.yaml').read_text())
    np.testing.assert_array_equal(experiment.canonical_path('E1_lateral', config),
                                  experiment.canonical_path('E1_response_L0p33_T0p5', config))
    ramp = experiment.canonical_path('E1_orientation_quarter', config)
    np.testing.assert_allclose(ramp[[0, 100, 115, 130, -1]],
        [[0, 0, 0], [1, 0, 0], [1.15, 0, np.pi/4], [1.3, 0, np.pi/2], [2.5, 0, np.pi/2]])
