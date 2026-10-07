"""Unscored positioning aims inside, and still verifies, the frozen start limits."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import dwvp_access_experiment as experiment
import dwvp_access_batch as batch
from test_access_batch import endpoint_miss


@pytest.mark.parametrize('changed', [None, 'general_goal_checker', 'alignment_goal_checker'])
def test_alignment_checker_keeps_scored_tuning_and_rejects_runtime_changes(tmp_path, changed):
    config = experiment.render_parameters(ROOT/'params/hsrb_dwvp_access_params.yaml', experiment.default_config())
    frozen = yaml.safe_dump(config)
    expected = tmp_path/'expected.yaml'; expected.write_text(frozen)
    actual = experiment.with_alignment_checker(config)
    assert yaml.safe_dump(config) == frozen
    server = actual['controller_server']['ros__parameters']
    checker = server[experiment.ALIGNMENT_GOAL_CHECKER]
    assert checker['xy_goal_tolerance'] == .05 and checker['yaw_goal_tolerance'] == .15
    assert server['general_goal_checker'] == config['controller_server']['ros__parameters']['general_goal_checker']
    if changed:
        server[changed]['xy_goal_tolerance'] = .2
    files = {}
    for name in ('controller_server', 'velocity_smoother'):
        files[name] = tmp_path/(name+'.yaml')
        files[name].write_text(yaml.safe_dump({'/'+name: actual[name]}))
    if changed:
        with pytest.raises(ValueError, match='Runtime parameter mismatch'):
            experiment.verify_runtime_parameters(expected, files, 'DWVP')
    else:
        for controller in experiment.CONTROLLERS:
            experiment.verify_runtime_parameters(expected, files, controller)
    with pytest.raises(ValueError, match='fixed target'):
        experiment.transfer_goal_checker(dict(goal_checker_id=experiment.ALIGNMENT_GOAL_CHECKER,
            endpoint_policy='reanchor_after_stop'), np.array([[0.,0.,0.], [1.,0.,0.]]))


@pytest.mark.parametrize('purpose,status,stopped,accepted', [
    ('reposition_only_not_an_experiment_trial',4,True,True),
    ('reposition_only_not_an_experiment_trial',6,True,False),
    ('reposition_only_not_an_experiment_trial',4,False,False),
    ('experiment_trial',4,True,False)])
def test_only_verified_positioning_miss_may_request_correction(monkeypatch,tmp_path,purpose,status,stopped,accepted):
    result = endpoint_miss(tmp_path)
    result.update(purpose=purpose, action_status=status)
    experiment.write_json(tmp_path/'result.json', result)
    child = NS(returncode=1, poll=lambda: 1)
    monkeypatch.setattr(batch.subprocess, 'Popen', lambda *a, **kw: child)
    monkeypatch.setattr(batch, 'spin_until', lambda *a: None)
    monkeypatch.setattr(batch, 'map_matches', lambda *a: True)
    def capture():
        if not stopped: raise RuntimeError('Stop not verified')
    observer = NS(stopped=capture,command_owners=lambda active: None,map=None)
    invoke = lambda: batch.recorded_child(observer, [], tmp_path/'log', tmp_path/'result.json', [],
        (None,None), positioning_manifest=dict(xy_tolerance_m=.1,yaw_tolerance_rad=.3))
    if accepted:
        assert invoke()['success'] is False
        assert json.loads((tmp_path/'result.json').read_text())['success'] is False
    else:
        with pytest.raises(RuntimeError): invoke()


def test_correction_targets_same_pose_and_checks_stopped_residual(tmp_path):
    manifest = dict(xy_tolerance_m=.1, yaw_tolerance_rad=.3)
    poses = iter([[0.,0.,0.], [.1062,0.,3.12], [.04,0.,3.12]])
    capture = lambda: dict(pose=next(poses))
    target = np.array([0.,0.,np.pi])
    attempts = []
    def move(current, attempt):
        attempts.append((current, attempt))
        return dict(action_status=4,success=attempt==2,final_pose_fresh=True,settling=dict(verified=True))
    report = batch.checked_fixed_alignment(target,manifest,capture,move,tmp_path/'checks/start.json')
    assert report['verified'] and len(attempts)==2
    assert report['attempts'][0]['result_success'] is False
    assert report['stopped']['pose']==[.04,0.,3.12]
    assert report['target']==target.tolist() and manifest['xy_tolerance_m']==.1


def test_correction_is_bounded_and_keeps_failed_attempts(tmp_path):
    calls = []
    def move(current, attempt):
        calls.append(attempt)
        return dict(action_status=4,success=False,final_pose_fresh=True,settling=dict(verified=True))
    record = tmp_path/'start.json'
    with pytest.raises(RuntimeError,match='after 3 attempts'):
        batch.checked_fixed_alignment([0,0,0],dict(xy_tolerance_m=.1,yaw_tolerance_rad=.3),
            lambda:dict(pose=[.1062,0,0]),move,record)
    assert calls==[1,2,3]
    report=json.loads(record.read_text())
    assert not report['verified'] and len(report['attempts'])==3


@pytest.mark.parametrize('failure', ['abort','stale','interrupt'])
def test_action_or_sensor_failures_never_start_a_correction(tmp_path,failure):
    calls=[]
    def capture():
        if calls and failure=='stale': raise RuntimeError('Fresh TF required')
        return dict(pose=[1,0,0])
    def move(current,attempt):
        calls.append(attempt)
        if failure=='interrupt': raise KeyboardInterrupt()
        return dict(action_status=6 if failure=='abort' else 4,success=False,
                    final_pose_fresh=True,settling=dict(verified=True))
    with pytest.raises((RuntimeError,KeyboardInterrupt)):
        batch.checked_fixed_alignment([0,0,0],dict(xy_tolerance_m=.1,yaw_tolerance_rad=.3),
            capture,move,tmp_path/'start.json')
    assert calls==[1]
