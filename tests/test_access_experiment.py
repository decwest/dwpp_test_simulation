import importlib.util
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('experiment', ROOT / 'scripts/dwvp_access_experiment.py')
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def test_balanced_reproducible_80_trial_schedule(tmp_path):
    params = ROOT / 'params/hsrb_dwvp_access_params.yaml'
    one = experiment.prepare(tmp_path / 'one', params)
    two = experiment.prepare(tmp_path / 'two', params)
    assert len(one['trials']) == len({t['id'] for t in one['trials']}) == 80
    assert one['trials'] == two['trials']
    counts = Counter((t['task'], t['controller']) for t in one['trials'])
    assert all(n == 5 for n in counts.values())
    assert {k[1] for k in counts if k[0].startswith('B_')} == {'MPPI', 'DWVP'}
    assert one['paths']['A_obstacle']['file'] is None
    assert experiment.summarize(tmp_path / 'one')['pending'] == 80


def test_pose_conditions_keep_position_geometry():
    for name in ('path1', 'path2'):
        common = experiment.canonical_path(name)
        independent = experiment.canonical_path(name, True)
        np.testing.assert_array_equal(common[:, :2], independent[:, :2])
        assert not np.allclose(common[:, 2], independent[:, 2])
    assert np.all(experiment.canonical_path('path1', True)[:, 2] == 0)
    assert experiment.canonical_path('path2', True)[-1, 2] == pytest.approx(np.pi / 2)
    path2 = experiment.canonical_path('path2')
    assert path2[-1, 0] == pytest.approx(1.5)
    assert path2[:, 1].max() == pytest.approx(1.5, abs=1e-3)


def test_freezes_map_origin_and_rejects_modified_inputs(tmp_path):
    params = ROOT / 'params/hsrb_dwvp_access_params.yaml'
    experiment.prepare(tmp_path / 'one', params, [3., 4., np.pi / 2])
    _, _, path = experiment.load_trial(tmp_path / 'one', 'B_path1_DWVP_r1')
    np.testing.assert_allclose(path[0], [3., 4., np.pi / 2])
    np.testing.assert_allclose(path[-1, :2], [2., 5.])
    with pytest.raises(ValueError, match='surveyed'):
        experiment.load_trial(tmp_path / 'one', 'A_obstacle_DWVP_r1')
    (tmp_path / 'one/nav2_params.yaml').write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        experiment.load_trial(tmp_path / 'one', 'B_path1_DWVP_r1')


def test_unset_origin_never_runs(tmp_path):
    experiment.prepare(tmp_path / 'one', ROOT / 'params/hsrb_dwvp_access_params.yaml')
    with pytest.raises(ValueError, match='origin'):
        experiment.load_trial(tmp_path / 'one', 'A_path1_RPP_r1')


def test_status_reports_unready_and_incomplete_trials_without_writing(tmp_path):
    root = tmp_path / 'session'
    experiment.prepare(root, ROOT / 'params/hsrb_dwvp_access_params.yaml', [0, 0, 0])
    before = set(root.rglob('*'))
    states = {r['trial']: r['status'] for r in experiment.session_status(root)}
    assert len(states) == 80 and states['A_obstacle_RPP_r1'] == 'needs_route'
    assert states['B_path1_DWVP_r1'] == 'pending'
    assert set(root.rglob('*')) == before
    folder = root / 'runs/B_path1_DWVP_r1'
    folder.mkdir(parents=True)
    states = {r['trial']: r['status'] for r in experiment.session_status(root)}
    assert states['B_path1_DWVP_r1'] == 'incomplete'


@pytest.mark.parametrize('source,received,frame,ranges,valid', [
    (10, 10, 'laser', [float('inf')] * 5, True),
    (10, 10, 'laser', [.8, float('nan')], True),
    (8, 10, 'laser', [.8], False),
    (10, 8, 'laser', [.8], False),
    (11, 10, 'laser', [.8], False),
    (10, 10, '', [.8], False),
    (10, 10, 'laser', [], False),
    (10, 10, 'laser', [float('nan')], False),
])
def test_laser_preflight_checks_source_time_and_usable_measurements(source, received, frame, ranges, valid):
    scan = SimpleNamespace(header=SimpleNamespace(frame_id=frame, stamp=SimpleNamespace(sec=source, nanosec=0)),
                           ranges=ranges, range_min=.05, range_max=5.)
    if valid:
        assert experiment.scan_quality(scan, received, 10.1)['frame'] == 'laser'
    else:
        with pytest.raises(RuntimeError):
            experiment.scan_quality(scan, received, 10.1)


def test_errors_use_same_projected_position_and_wrapped_yaw():
    path = np.array([[0., 0., np.deg2rad(170)], [2., 0., np.deg2rad(-170)]])
    errors = experiment.tracking_errors(np.array([[1., .2, np.pi]]), path)
    np.testing.assert_allclose(errors, [[.2, 0]], atol=1e-12)


def test_invalid_reference_rejected():
    with pytest.raises(ValueError, match='Duplicate'):
        experiment.validate_path([[0, 0, 0], [0, 0, 1]])
    with pytest.raises(ValueError, match='finite'):
        experiment.validate_path([[0, 0, 0], [1, 0, np.nan]])


def test_failed_runs_remain_in_counts_and_never_get_success_means(tmp_path):
    root = tmp_path / 'session'
    experiment.prepare(root, ROOT / 'params/hsrb_dwvp_access_params.yaml', [0, 0, 0])
    folder = root / 'runs/A_path1_RPP_r1'
    folder.mkdir(parents=True)
    (folder / 'result.json').write_text(json.dumps({'status': 'timeout', 'success': False, 'duration_s': 120}))
    (folder / 'tracking.csv').write_text('t,stamp_s,x,y,yaw,tf_age_s,raw_age_s,applied_age_s,odom_age_s,odom_source_age_s\n0,1,0,0,0,0,0,0,0,0\n')
    np.savetxt(folder / 'reference.csv', experiment.canonical_path('path1'), delimiter=',', header='x,y,yaw', comments='')
    report = experiment.summarize(root)
    assert report['recorded'] == 1 and report['pending'] == 79
    assert report['groups'][0]['succeeded'] == 0
    assert report['groups'][0]['duration_s_mean'] is None


def test_command_diagnostics_keep_receipt_jitter_separate(tmp_path):
    file = tmp_path / 'commands.csv'
    file.write_text('stamp_s,source_stamp_s,vx,vy,omega\n1,,0,0,0\n1.031,,0.007333333333333333,0,0\n1.070,,0.014666666666666666,0,0\n')
    result = experiment.command_metrics(file, 1, .1, 30)
    assert result['command_increment_excess_samples'] == 0
    assert result['velocity_excess_samples'] == 0
    assert result['receive_interval_max_s'] == pytest.approx(.039)
    assert result['linear_command_jerk_rms_m_s3'] < 1e-10


def test_runtime_configuration_mismatch_is_detected(tmp_path):
    expected = ROOT / 'params/hsrb_dwvp_access_params.yaml'
    config = yaml.safe_load(expected.read_text())
    runtime_files = {}
    for name in ('controller_server', 'velocity_smoother'):
        file = tmp_path / (name + '.yaml')
        file.write_text(yaml.safe_dump({'/' + name: config[name]}))
        runtime_files[name] = file
    experiment.verify_runtime_parameters(expected, runtime_files, 'DWVP')
    config['velocity_smoother']['ros__parameters']['max_accel'] = [99., 99., 99.]
    runtime_files['velocity_smoother'].write_text(yaml.safe_dump({'/velocity_smoother': config['velocity_smoother']}))
    with pytest.raises(ValueError, match='Runtime parameter mismatch'):
        experiment.verify_runtime_parameters(expected, runtime_files, 'DWVP')


@pytest.mark.parametrize('controller,field,value', [
    ('DWVP', 'lookahead_time', 99.),
    ('DWVP', 'min_vel_y', -.9),
    ('DWVP', 'approach_velocity_scaling_dist', 0.),
    ('DWVP', 'min_orientation_time', 8.),
    ('RPP', 'desired_linear_vel', 9.),
    ('DWPP', 'max_linear_accel', 9.),
    ('MPPI', 'PathAlignCritic', {'enabled': False}),
])
def test_selected_controller_tuning_is_frozen(tmp_path, controller, field, value):
    expected = ROOT / 'params/hsrb_dwvp_access_params.yaml'
    config = yaml.safe_load(expected.read_text())
    config['controller_server']['ros__parameters'][controller][field] = value
    files = {}
    for name in ('controller_server', 'velocity_smoother'):
        files[name] = tmp_path / (name + '.yaml')
        files[name].write_text(yaml.safe_dump({'/' + name: config[name]}))
    with pytest.raises(ValueError, match='Runtime parameter mismatch'):
        experiment.verify_runtime_parameters(expected, files, controller)


@pytest.mark.parametrize('missing_prefix,source_age,expected_valid', [
    (.02, 0., 1), (.4, 0., 0), (.02, 10., 0), (.02, -.5, 0),
])
def test_summary_retains_prefix_and_rejects_stale_source(tmp_path, missing_prefix, source_age, expected_valid):
    root = tmp_path / 'session'
    experiment.prepare(root, ROOT / 'params/hsrb_dwvp_access_params.yaml', [0, 0, 0])
    folder = root / 'runs/B_path1_DWVP_r1'
    folder.mkdir(parents=True)
    (folder / 'result.json').write_text(json.dumps({'status': 'succeeded', 'success': True, 'duration_s': 1.,
                                                  'missing_command_prefix_s': missing_prefix}))
    (folder / 'tracking.csv').write_text(
        't,stamp_s,x,y,yaw,tf_age_s,raw_age_s,applied_age_s,odom_age_s,odom_source_age_s\n'
        f'.005,1,0,0,0,0,inf,inf,0,{source_age}\n'
        f'.5,1.5,0,0,0,0,0,0,0,{source_age}\n')
    np.savetxt(folder / 'reference.csv', experiment.canonical_path('path1', True), delimiter=',', header='x,y,yaw', comments='')
    report = experiment.summarize(root)
    assert report['trials'][0]['samples'] == 2
    assert report['trials'][0]['missing_command_prefix_s'] == missing_prefix
    assert report['trials'][0]['stale_or_missing_samples'] >= 1
    assert report['groups'][0]['valid_successes'] == expected_valid
