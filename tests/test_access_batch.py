"""Geometry and schedule checks without a ROS graph."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import dwvp_access_experiment as experiment
import dwvp_access_batch as batch


@pytest.mark.parametrize('current', [[0, 0, 0], [1.2, -3.1, 1.3], [2, 1, -3.1]])
def test_measured_start_preserves_lateral_error_for_all_conditions(tmp_path, current):
    session = tmp_path / 'session'
    manifest = experiment.prepare(session, ROOT / 'params/hsrb_dwvp_access_params.yaml', current_start=current)
    for name in experiment.CONDITIONS[:-1]:
        _, trial, path = experiment.load_trial(session, name + '_DWVP_r1')
        np.testing.assert_allclose(experiment.start_pose(manifest, trial), current, atol=1e-12)
        origin = manifest['map_origins'][name]
        np.testing.assert_allclose(experiment.transform_poses([[0, 0, 0]], origin)[0], path[0], atol=1e-12)
        distance = np.linalg.norm(path[0, :2] - current[:2])
        assert distance == pytest.approx(.5 if name == 'E1_lateral' else 0.)
    lateral = experiment.load_trial(session, 'E1_lateral_DWVP_r1')[2]
    left = np.array([-np.sin(current[2]), np.cos(current[2])])
    assert np.dot(np.array(current[:2])-lateral[0, :2], left) == pytest.approx(.5)


@pytest.mark.parametrize('condition', experiment.CONDITIONS[:-1])
def test_each_trial_can_reanchor_without_changing_local_experimental_geometry(tmp_path, condition):
    session = tmp_path/'session'
    manifest = experiment.prepare(session, ROOT/'params/hsrb_dwvp_access_params.yaml',
        current_start=[0, 0, 0], bidirectional=True, conditions=[condition], repeats=[1])
    measured = np.array([1.3, -2.1, 2.9])
    for trial in manifest['trials'][:2]:
        _, _, original = experiment.load_trial(session, trial['id'])
        placed, path = experiment.reanchor_trial(manifest, trial, original, measured)
        np.testing.assert_allclose(experiment.start_pose(manifest, placed), measured)
        np.testing.assert_allclose(np.linalg.norm(np.diff(path[:,:2],axis=0),axis=1),
                                   np.linalg.norm(np.diff(original[:,:2],axis=0),axis=1))
        expected_local = experiment.transform_poses(original, experiment.origin_from_start([0,0,0], experiment.start_pose(manifest,trial)))
        measured_local = experiment.transform_poses(path, experiment.origin_from_start([0,0,0], measured))
        np.testing.assert_allclose(measured_local, expected_local, atol=1e-12)
        assert trial['start_pose'] != placed['start_pose']


def test_current_start_metrics_and_resume_use_verified_recorded_geometry(tmp_path):
    session = tmp_path/'session'
    manifest = experiment.prepare(session, ROOT/'params/hsrb_dwvp_access_params.yaml',
        current_start=[0,0,0], bidirectional=True, conditions=['E1_lateral'], repeats=[1])
    # Resume can cross from older fixed-placement successes into reanchored ones.
    old_trial, trial = manifest['trials'][:2]
    old_folder = write_success(session, old_trial)
    _, _, old_path = experiment.load_trial(session, old_trial['id'])
    np.savetxt(old_folder/'reference.csv', old_path, delimiter=',', header='x,y,yaw', comments='')
    old_bytes = {p.name: p.read_bytes() for p in old_folder.iterdir()}
    _, _, frozen = experiment.load_trial(session, trial['id'])
    measured = [1.3, -2.1, 1.2]
    placed, reference = experiment.reanchor_trial(manifest, trial, frozen, measured)
    folder = write_success(session, trial)
    np.savetxt(folder/'reference.csv', reference, delimiter=',', header='x,y,yaw', comments='')
    experiment.write_json(folder/'start_capture.json', dict(capture=dict(pose=measured)))
    meta = json.loads((folder/'trial.json').read_text())
    meta.update(reference_policy='per_trial_current_pose', reference_sha256=experiment.digest(folder/'reference.csv'),
                start_capture_sha256=experiment.digest(folder/'start_capture.json'))
    experiment.write_json(folder/'trial.json', meta)
    experiment.write_json(folder/'result.json', dict(status='succeeded', success=True, duration_s=1.,
        missing_command_prefix_s=0., reference_policy='per_trial_current_pose'))
    (folder/'tracking.csv').write_text('t,stamp_s,x,y,yaw,tf_age_s,raw_age_s,applied_age_s,odom_age_s,odom_source_age_s\n'
        +','.join(map(str,[0,1,*measured,0,0,0,0,0]))+'\n')
    row = next(t for t in experiment.summarize(session)['trials'] if t['trial_id']==trial['id'])
    assert row['lateral_initial_error_m'] == pytest.approx(.5)
    assert row['reference_policy'] == 'per_trial_current_pose'
    assert batch.selected_trials(session, resume=True)[1] == manifest['trials'][2:]
    assert old_bytes == {name: (old_folder/name).read_bytes() for name in old_bytes}
    with (folder/'reference.csv').open('a') as stream: stream.write('9,9,9\n')
    with pytest.raises(ValueError, match='changed'):
        batch.selected_trials(session, resume=True)


def test_legacy_manual_origin_remains_common_and_invalid_anchor_reserves_nothing(tmp_path):
    params = ROOT / 'params/hsrb_dwvp_access_params.yaml'
    manifest = experiment.prepare(tmp_path/'legacy', params, [1., 2., 0.])
    assert 'map_origins' not in manifest
    np.testing.assert_allclose(manifest['starts']['E1_lateral']['map_pose'], [1, 2.5, 0])
    for pose in ([0, float('nan'), 0], [1, 2]):
        with pytest.raises(ValueError):
            experiment.prepare(tmp_path/'invalid', params, current_start=pose)
        assert not (tmp_path/'invalid').exists()
    with pytest.raises(ValueError):
        experiment.prepare(tmp_path/'invalid', params, [0, 0, 0], current_start=[0, 0, 0])


def test_missing_e2_is_not_silently_skipped_and_repeat_order_is_frozen(tmp_path):
    manifest = experiment.prepare(tmp_path/'session', ROOT/'params/hsrb_dwvp_access_params.yaml', current_start=[0, 0, 0])
    with pytest.raises(ValueError, match='Supply the E2 route'):
        batch.selected_trials(tmp_path/'session')
    _, trials = batch.selected_trials(tmp_path/'session', ['E1'])
    assert len(trials) == 55
    assert trials == [t for t in manifest['trials'] if t['task'] != 'E2_environment']
    _, first = batch.selected_trials(tmp_path/'session', ['E1'], [1])
    assert len(first) == 11
    reserved = tmp_path/'session'/'runs'/first[0]['id']
    reserved.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        batch.selected_trials(tmp_path/'session', ['E1'], [1])


def test_e2_keeps_map_coordinates_when_e1_is_anchored(tmp_path):
    route = tmp_path/'route.csv'
    path = np.array([[3., 4., .2], [4., 4.2, .2]])
    np.savetxt(route, path, delimiter=',', header='x,y,yaw', comments='')
    experiment.prepare(tmp_path/'session', ROOT/'params/hsrb_dwvp_access_params.yaml',
                       current_start=[1, 2, 1.5], environment_path=route)
    manifest, trial, actual = experiment.load_trial(tmp_path/'session', 'E2_environment_DWVP_r1')
    np.testing.assert_allclose(actual, path)
    np.testing.assert_allclose(experiment.start_pose(manifest, trial), path[0])
    assert len(batch.selected_trials(tmp_path/'session')[1]) == 80


def write_success(session, trial):
    folder = session/'runs'/trial['id']; folder.mkdir(parents=True)
    experiment.write_json(folder/'result.json', dict(status='succeeded', success=True))
    experiment.write_json(folder/'trial.json', dict(trial=trial,
        manifest_sha256=experiment.digest(session/'manifest.json'), params_sha256=trial['params_sha256']))
    return folder


def endpoint_miss(folder, *, legacy=False):
    result = dict(status='failed', success=False, action_status=4, endpoint_policy='fixed_tolerance',
                  final_pose_within_tolerances=False, final_pose=[1., 0., .52],
                  final_position_error_m=.0895, final_yaw_error_rad=.5196,
                  settling=dict(verified=True),
                  error='Final stopped pose outside tolerances or action unsuccessful: action_status=4, yaw=0.5196')
    if legacy:
        (folder/'settling.csv').write_text('x,y,yaw,fresh,applied_zero,speed_m_s,yaw_rate_rad_s\n'
                                         '1,0,.52,True,True,0,.001\n')
    else:
        result.update(failure_reason='endpoint_tolerance_exceeded', final_pose_fresh=True)
    experiment.write_json(folder/'result.json', result)
    return result


@pytest.mark.parametrize('legacy', [False, True])
def test_resume_keeps_endpoint_failure_and_remaining_frozen_order(tmp_path, legacy):
    session = tmp_path/'session'
    manifest = experiment.prepare(session, ROOT/'params/hsrb_dwvp_access_params.yaml',
        current_start=[0,0,0], bidirectional=True, conditions=['E1'], repeats=[1])
    write_success(session, manifest['trials'][0])
    folder = write_success(session, manifest['trials'][1])
    endpoint_miss(folder, legacy=legacy)
    original = {str(p): p.read_bytes() for p in session.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='failed trial'):
        batch.selected_trials(session, resume=True)
    _, pending = batch.selected_trials(session, resume=True, continue_on_endpoint_failure=True)
    assert pending == manifest['trials'][2:]
    assert {str(p): p.read_bytes() for p in session.rglob('*') if p.is_file()} == original
    assert json.loads((folder/'result.json').read_text())['success'] is False
    # Relaxing continuation never relaxes frozen input verification.
    meta = json.loads((folder/'trial.json').read_text()); meta['params_sha256'] = 'wrong'
    experiment.write_json(folder/'trial.json', meta)
    with pytest.raises(ValueError, match='Recorded inputs differ'):
        batch.selected_trials(session, resume=True, continue_on_endpoint_failure=True)


@pytest.mark.parametrize('change', [
    dict(action_status=6), dict(action_status=5), dict(status='timeout'), dict(status='error'),
    dict(settling=dict(verified=False)), dict(final_pose_fresh=False),
    dict(final_yaw_error_rad=float('nan')), dict(final_pose=[float('nan'),0,0]),
    dict(final_position_error_m=.02, final_yaw_error_rad=.03),
    dict(endpoint_policy='reanchor_after_stop'), dict(error='Fresh TF required'),
    dict(failure_reason='sensor_failure'),
])
def test_only_stopped_endpoint_misses_may_continue(tmp_path, change):
    result = endpoint_miss(tmp_path)
    result.update(change)
    assert not batch.settled_endpoint_failure(result, dict(xy_tolerance_m=.1, yaw_tolerance_rad=.3), tmp_path)


def test_legacy_resume_requires_fresh_zero_command_and_matching_final_sample(tmp_path):
    result = endpoint_miss(tmp_path, legacy=True)
    manifest = dict(xy_tolerance_m=.1, yaw_tolerance_rad=.3)
    file = tmp_path/'settling.csv'
    original = file.read_text()
    for contents in ('', original.replace('True,True', 'False,True'),
                     original.replace('True,True', 'True,False'), original.replace('.52', '.6')):
        file.write_text(contents)
        assert not batch.settled_endpoint_failure(result, manifest, tmp_path)
    file.unlink()
    assert not batch.settled_endpoint_failure(result, manifest, tmp_path)


@pytest.mark.parametrize('allow,exit_code,stopped,expected', [
    (False,0,True,False), (True,0,True,True), (True,1,True,False), (True,0,False,False)])
def test_recorder_continuation_requires_clean_exit_and_live_stopped_pose(monkeypatch, tmp_path,
                                                                       allow, exit_code, stopped, expected):
    endpoint_miss(tmp_path)
    manifest = dict(xy_tolerance_m=.1, yaw_tolerance_rad=.3)
    child = NS(returncode=exit_code, poll=lambda: exit_code)
    monkeypatch.setattr(batch.subprocess, 'Popen', lambda *a, **kw: child)
    monkeypatch.setattr(batch, 'spin_until', lambda observer, pred, timeout, guard: None)
    monkeypatch.setattr(batch, 'map_matches', lambda *a: True)
    def verify_stop():
        if not stopped: raise RuntimeError('Fresh stopped pose unavailable')
    observer = NS(stopped=verify_stop, command_owners=lambda active: None, map=None)
    kwargs = dict(endpoint_manifest=manifest) if allow else {}
    if expected:
        assert batch.recorded_child(observer, [], tmp_path/'log', tmp_path/'result.json', [], (None,None), **kwargs)['success'] is False
    else:
        with pytest.raises(RuntimeError):
            batch.recorded_child(observer, [], tmp_path/'log', tmp_path/'result.json', [], (None,None), **kwargs)


def test_resume_preserves_successful_prefix_and_reverse_start(tmp_path):
    session = tmp_path/'session'
    manifest = experiment.prepare(session, ROOT/'params/hsrb_dwvp_access_params.yaml',
        current_start=[0, 0, 0], bidirectional=True, conditions=['E1'], repeats=[1])
    folder = write_success(session, manifest['trials'][0])
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    _, trials = batch.selected_trials(session, resume=True)
    assert trials == manifest['trials'][1:] and trials[0]['direction'] == 'reverse'
    assert before == {p.name: p.read_bytes() for p in folder.iterdir()}
    for trial in trials: write_success(session, trial)
    assert batch.selected_trials(session, resume=True)[1] == []


@pytest.mark.parametrize('damage', ['incomplete', 'failed', 'changed', 'gap'])
def test_resume_refuses_failed_missing_or_changed_history(tmp_path, damage):
    session = tmp_path/'session'
    manifest = experiment.prepare(session, ROOT/'params/hsrb_dwvp_access_params.yaml',
        current_start=[0, 0, 0], bidirectional=True, conditions=['E1'], repeats=[1])
    trial = manifest['trials'][1 if damage=='gap' else 0]
    folder = write_success(session, trial)
    if damage == 'incomplete': (folder/'result.json').unlink()
    if damage == 'failed': experiment.write_json(folder/'result.json', dict(status='failed', success=False))
    if damage == 'changed': experiment.write_json(folder/'trial.json', dict(trial=trial, manifest_sha256='wrong'))
    with pytest.raises(ValueError): batch.selected_trials(session, resume=True)


def test_return_keeps_planner_detour_and_wraps_yaw():
    xy = [[2, 0], [2, 1], [0, 1], [0, 0]]
    path = batch.relocation_path(xy, [2, 0, np.deg2rad(179)], [0, 0, np.deg2rad(-179)])
    np.testing.assert_allclose(path[:, :2], xy)
    assert max(abs(np.diff(np.unwrap(path[:, 2])))) < np.deg2rad(2)
    assert path[-1, 2] == pytest.approx(np.deg2rad(-179))
    rotation = batch.relocation_path([[0, 0]], [0, 0, 0], [0, 0, 1])
    np.testing.assert_allclose(rotation, [[0, 0, 0], [0, 0, 1]])


def test_live_map_must_match_unknowns_and_geometry():
    info = dict(resolution=.05, origin=[-1., 2., 0.], negate=0, free_thresh=.196, occupied_thresh=.65)
    pixels = np.array([[205, 254], [0, 254]], dtype=np.uint8)
    msg = NS(header=NS(frame_id='map'), data=[100, 0, -1, 0], info=NS(
        width=2, height=2, resolution=.05, origin=NS(position=NS(x=-1., y=2.),
        orientation=NS(x=0., y=0., z=0., w=1.))))
    assert batch.map_matches(msg, info, pixels)
    msg.data[2] = 0
    assert not batch.map_matches(msg, info, pixels)
    msg.data[2] = -1; msg.info.origin.position.x = 0
    assert not batch.map_matches(msg, info, pixels)


def test_geometry_rejects_obstacle_before_any_motion(tmp_path):
    from PIL import Image
    session = tmp_path/'session'
    experiment.prepare(session, ROOT/'params/hsrb_dwvp_access_params.yaml', current_start=[0, 0, 0])
    _, trials = batch.selected_trials(session, ['E1'])
    image = np.full((200, 200), 254, dtype=np.uint8)
    Image.fromarray(image).save(tmp_path/'map.pgm')
    info = dict(image='map.pgm', resolution=.05, origin=[-5., -5., 0.], negate=0,
                free_thresh=.196, occupied_thresh=.65, mode='trinary')
    (tmp_path/'map.yaml').write_text(yaml.safe_dump(info))
    batch.check_geometry(session, trials, tmp_path/'map.yaml')
    image[109, 120] = 0  # map (1.025, -0.475), on the lateral reference
    Image.fromarray(image).save(tmp_path/'map.pgm')
    with pytest.raises(ValueError, match='intersects'):
        batch.check_geometry(session, trials, tmp_path/'map.yaml')


def test_reanchored_geometry_is_checked_at_its_new_location(tmp_path):
    from PIL import Image
    session = tmp_path/'session'
    manifest = experiment.prepare(session, ROOT/'params/hsrb_dwvp_access_params.yaml',
        current_start=[-1,0,0], bidirectional=True, conditions=['E1_lateral'], repeats=[1])
    trial = manifest['trials'][0]
    image = np.full((200,200),254,dtype=np.uint8); image[89,110] = 0
    Image.fromarray(image).save(tmp_path/'map.pgm')
    (tmp_path/'map.yaml').write_text(yaml.safe_dump(dict(image='map.pgm', resolution=.05,
        origin=[-5,-5,0], mode='trinary', negate=0, occupied_thresh=.65, free_thresh=.196)))
    data = batch.check_geometry(session, [trial], tmp_path/'map.yaml')
    _, _, reference = experiment.load_trial(session, trial['id'])
    placed, path = experiment.reanchor_trial(manifest, trial, reference, [-1,1,0])
    with pytest.raises(ValueError, match='intersects'):
        batch.check_placed_geometry(path, experiment.start_pose(manifest,placed), session/trial['params_file'], data)
    with pytest.raises(ValueError, match='E1 only'):
        experiment.reanchor_trial(manifest, dict(trial,task='E2_environment'), reference, [0,0,0])


@pytest.mark.parametrize('condition', experiment.CONDITIONS[:-1])
def test_reverse_leg_preserves_the_experimental_initial_condition(tmp_path, condition):
    initial = [1., -2., .7]
    session = tmp_path/'session'
    manifest = experiment.prepare(session, ROOT/'params/hsrb_dwvp_access_params.yaml',
        current_start=initial, bidirectional=True, conditions=[condition], repeats=[1])
    first, second = manifest['trials'][:2]
    assert [first['direction'], second['direction']] == ['forward', 'reverse']
    _, _, forward = experiment.load_trial(session, first['id'])
    _, _, reverse = experiment.load_trial(session, second['id'])
    far_start = experiment.start_pose(manifest, second)
    np.testing.assert_allclose(far_start[:2], forward[-1, :2], atol=1e-12)
    np.testing.assert_allclose(reverse[-1, :2], initial[:2], atol=1e-12)
    assert abs(float(experiment.wrap(far_start[2]-initial[2]))) == pytest.approx(np.pi)
    direction = (reverse[-1, :2]-reverse[0, :2])/2.5
    normal = np.array([-direction[1], direction[0]])
    assert (far_start[:2]-reverse[0, :2]) @ normal == pytest.approx(.5 if condition=='E1_lateral' else 0.)
    assert float(experiment.wrap(reverse[-1, 2]-reverse[0, 2])) == pytest.approx(0 if condition=='E1_lateral' else np.pi/2)


def test_bidirectional_selection_and_directional_denominators(tmp_path):
    session = tmp_path/'session'
    manifest = experiment.prepare(session, ROOT/'params/hsrb_dwvp_access_params.yaml',
        current_start=[0, 0, 0], bidirectional=True, conditions=['E1'], repeats=[1])
    assert len(manifest['trials']) == 11
    assert [t['direction'] for t in manifest['trials']] == ['forward','reverse']*5+['forward']
    report = experiment.summarize(session)
    assert sum(g['planned'] for g in report['groups']) == 11
    assert all(g['planned'] == 1 for g in report['groups'])
    assert batch.selected_trials(session)[1] == manifest['trials']
    reverse_trial = manifest['trials'][1]
    # Selecting only the reverse controller's condition/iteration can no longer
    # pretend the robot is at the near-side captured start.
    manifest['trials'] = [reverse_trial]
    (session/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='alternation'):
        batch.selected_trials(session)


def test_e2_reverse_is_map_fixed_with_opposite_tangent(tmp_path):
    path = np.array([[2, 3, .1], [3, 3.2, .2], [4, 3.4, .2]])
    route = tmp_path/'route.csv'
    np.savetxt(route, path, delimiter=',', header='x,y,yaw', comments='')
    session = tmp_path/'session'
    manifest = experiment.prepare(session, ROOT/'params/hsrb_dwvp_access_params.yaml',
        current_start=[0, 0, 0], environment_path=route, bidirectional=True,
        conditions=['E2_environment'], repeats=[1])
    second = manifest['trials'][1]
    _, _, reverse = experiment.load_trial(session, second['id'])
    np.testing.assert_allclose(reverse[:, :2], path[::-1, :2])
    np.testing.assert_allclose(np.cos(reverse[:, 2]), -np.cos(path[::-1, 2]))
    np.testing.assert_allclose(experiment.start_pose(manifest, second), reverse[0])
