"""Retries use the current robot pose, keeping map checks and local conditions."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import dwvp_access_batch as batch
import dwvp_access_experiment as experiment
from dwvp_access_path import PathBlockedError, assert_clear, load_map


def setup_case(tmp_path, original_start=(0,0,0)):
    session=tmp_path/'session'
    manifest=experiment.prepare(session,ROOT/'params/hsrb_dwvp_access_params.yaml',
        current_start=list(original_start),bidirectional=True,conditions=['E1_orientation_quarter'],repeats=[1])
    trial=manifest['trials'][0]
    _,_,reference=experiment.load_trial(session,trial['id'])
    pixels=np.full((100,100),254,dtype=np.uint8)
    pixels[99-28,70]=0  # map rectangle x=[2.5,2.55], y=[.4,.45]
    Image.fromarray(pixels).save(tmp_path/'map.pgm')
    (tmp_path/'map.yaml').write_text(yaml.safe_dump(dict(image='map.pgm',resolution=.05,
        origin=[-1,-1,0],mode='trinary',negate=0,occupied_thresh=.65,free_thresh=.196)))
    return manifest,trial,reference,session/trial['params_file'],load_map(tmp_path/'map.yaml')


def no_alignment(target):
    pytest.fail('Retry must never move to or turn toward the original start')


def test_retry_ignores_old_start_and_path_even_when_outside_map(tmp_path):
    manifest,trial,path,params,data=setup_case(tmp_path,original_start=[6,6,1.2])
    with pytest.raises(PathBlockedError):
        batch.check_placed_geometry(path,experiment.start_pose(manifest,trial),params,data)
    before=params.read_bytes()
    current=[0,0,.02]
    captured,placed,actual=batch.checked_current_placement(manifest,trial,path,params,data,
        lambda:dict(pose=current),no_alignment,tmp_path/'checks',current_only=True)
    np.testing.assert_allclose(experiment.start_pose(manifest,placed),current)
    np.testing.assert_allclose(actual,experiment.reanchor_trial(manifest,trial,path,current)[1])
    np.testing.assert_allclose(experiment.arclength(actual[:,:2])[-1],2.5)
    assert params.read_bytes()==before
    record=json.loads(next((tmp_path/'checks').glob('*.json')).read_text())
    assert record['static_map_checked'] and record['alignment_target'] is None
    assert record['start_policy']=='current_pose_with_turnaround'


def test_blocked_current_path_stops_without_trying_any_alignment(tmp_path):
    inputs=setup_case(tmp_path)
    with pytest.raises(RuntimeError,match='current pose is blocked'):
        batch.checked_current_placement(*inputs,lambda:dict(pose=[0,0,.086]),no_alignment,
                                         tmp_path/'checks',current_only=True)
    record=json.loads(next((tmp_path/'checks').glob('*.json')).read_text())
    assert not record['static_map_checked'] and record['alignment_target'] is None
    assert record['collision']['cell_index']==[70,28]
    assert record['collision']['distance_to_cell_m']<=.22
    assert not (tmp_path/'session/runs').exists()


def test_normal_batch_still_aligns_then_uses_measured_pose(tmp_path):
    inputs=setup_case(tmp_path)
    calls=[]
    current=[0,0,.02]
    captured,placed,path=batch.checked_current_placement(*inputs,lambda:dict(pose=current),
        lambda target:calls.append(target.tolist()),tmp_path/'checks')
    assert calls==[[0.,0.,0.]]
    np.testing.assert_allclose(experiment.start_pose(inputs[0],placed),current)
    assert json.loads(next((tmp_path/'checks').glob('*.json')).read_text())['start_policy']=='align_then_current_pose'


def test_each_retry_captures_again_instead_of_reusing_first_start(tmp_path):
    inputs=setup_case(tmp_path)
    for i,pose in enumerate(([.5,.5,.3],[.5,1.,0.])):
        captured,placed,path=batch.checked_current_placement(*inputs,lambda:dict(pose=pose),
            no_alignment,tmp_path/f'checks_{i}',current_only=True)
        np.testing.assert_allclose(experiment.start_pose(inputs[0],placed),pose)
        np.testing.assert_allclose(path[0,:2],pose[:2])


def test_sensor_failure_never_uses_old_pose_as_fallback(tmp_path):
    inputs=setup_case(tmp_path)
    def stale(): raise RuntimeError('Fresh TF required')
    with pytest.raises(RuntimeError,match='Fresh TF'):
        batch.checked_current_placement(*inputs,stale,no_alignment,tmp_path/'checks',current_only=True)
    assert not list((tmp_path/'checks').iterdir())


def test_collision_details_locate_the_map_cell(tmp_path):
    manifest,trial,path,params,data=setup_case(tmp_path)
    _,placed=experiment.reanchor_trial(manifest,trial,path,[0,0,.086])
    with pytest.raises(PathBlockedError) as error:
        assert_clear(placed,data[0],data[2],.22)
    hit=error.value.details
    np.testing.assert_allclose(hit['cell_map'],[2.525,.425])
    assert hit['distance_to_cell_m']<=.22 and not hit['outside_map']


@pytest.mark.parametrize('heading', [0., .4, 3.1, -3.1])
def test_turnaround_reverses_travel_not_final_body_yaw(tmp_path, heading):
    manifest,trial,path,params,data=setup_case(tmp_path)
    _,previous=experiment.reanchor_trial(manifest,trial,path,[0.,0.,heading])
    next_trial=manifest['trials'][1]
    next_path=experiment.load_trial(params.parent,next_trial['id'])[2]
    yaw=batch.turnaround_yaw(previous,next_path,experiment.start_pose(manifest,next_trial))
    _,actual=experiment.reanchor_trial(manifest,next_trial,next_path,[0.,0.,yaw])
    axis=actual[-1,:2]-actual[0,:2]
    prior=previous[-1,:2]-previous[0,:2]
    assert np.dot(axis,prior)/(np.linalg.norm(axis)*np.linalg.norm(prior))==pytest.approx(-1.)
    # This condition changes the body's yaw by 90 degrees while driving straight.
    assert abs(experiment.wrap(yaw-(previous[-1,2]+np.pi)))==pytest.approx(np.pi/2)


def recorded_predecessor(tmp_path, previous_start=(0.,0.,0.)):
    manifest,first,path,params,data=setup_case(tmp_path,original_start=[6,6,1.2])
    session=params.parent
    folder=session/'runs'/first['id']; folder.mkdir(parents=True)
    _,path=experiment.reanchor_trial(manifest,first,path,previous_start)
    np.savetxt(folder/'reference.csv',path,delimiter=',',header='x,y,yaw',comments='')
    experiment.write_json(folder/'start_capture.json',dict(capture=dict(pose=list(previous_start))))
    experiment.write_json(folder/'trial.json',dict(trial=first,
        reference_policy='per_trial_current_pose',manifest_sha256=experiment.digest(session/'manifest.json'),
        reference_sha256=experiment.digest(folder/'reference.csv'),
        start_capture_sha256=experiment.digest(folder/'start_capture.json')))
    experiment.write_json(folder/'result.json',dict(success=True,status='succeeded',final_pose=[2.4,0.,np.pi/2]))
    next_trial=manifest['trials'][1]
    next_path=experiment.load_trial(session,next_trial['id'])[2]
    return session,manifest,next_trial,next_path,session/next_trial['params_file'],data


def test_first_retry_needs_no_turn_or_previous_record(tmp_path):
    manifest,trial,path,params,data=setup_case(tmp_path)
    assert batch.checked_retry_turnaround(params.parent,manifest,trial,path,params,data,
        lambda:pytest.fail('First retry does not need a turnaround'),no_alignment,tmp_path/'turns') is None


def test_resume_turns_at_current_xy_using_recorded_path_and_is_idempotent(tmp_path):
    inputs=recorded_predecessor(tmp_path)
    state=dict(pose=[2.4,0.,np.pi/2])
    calls=[]
    def turn(target):
        calls.append(target.tolist())
        state['pose']=target.tolist()
    for attempt in range(2):
        record=batch.checked_retry_turnaround(*inputs,lambda:dict(state),turn,tmp_path/f'turns_{attempt}',
                                              resuming_first=True)
        assert record['stopped_heading_verified'] and record['static_map_checked']
    np.testing.assert_allclose(calls,[[2.4,0.,-np.pi],[2.4,0.,-np.pi]],atol=1e-10)


def test_blocked_return_direction_does_not_send_turn_goal(tmp_path):
    inputs=recorded_predecessor(tmp_path)
    with pytest.raises(RuntimeError,match='no turnaround goal sent'):
        batch.checked_retry_turnaround(*inputs,lambda:dict(pose=[2.4,.5,np.pi/2]),
            no_alignment,tmp_path/'turns')
    record=json.loads(next((tmp_path/'turns').glob('*.json')).read_text())
    assert 'collision' in record and not record['static_map_checked']


def test_incomplete_turn_does_not_start_next_trial(tmp_path):
    inputs=recorded_predecessor(tmp_path)
    calls=[]
    with pytest.raises(RuntimeError,match='no trial goal sent'):
        batch.checked_retry_turnaround(*inputs,lambda:dict(pose=[2.4,0.,np.pi/2]),
            lambda target:calls.append(target.tolist()),tmp_path/'turns')
    assert len(calls)==batch.TURNAROUND_MAX_ATTEMPTS


def test_changed_predecessor_geometry_cannot_choose_return_direction(tmp_path):
    inputs=recorded_predecessor(tmp_path)
    first=inputs[1]['trials'][0]
    path=inputs[0]/'runs'/first['id']/'reference.csv'
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError,match='geometry or capture changed'):
        batch.checked_retry_turnaround(*inputs,lambda:dict(pose=[2.4,0.,np.pi/2]),
            no_alignment,tmp_path/'turns')


def wall_near_return_endpoint(tmp_path):
    inputs=list(recorded_predecessor(tmp_path,previous_start=[.1437313335,-.3182716484,-.0196300537]))
    info=dict(image='unused.pgm',resolution=.05,origin=[-1.38,-4.11,0.])
    blocked=np.zeros((200,100),dtype=bool)
    # Cells from the actual failure; one blocks the measured yaw, the other
    # still blocks the ideal reverse heading after the 2 cm position change.
    blocked[71:73,24]=True
    inputs[-1]=(info,np.where(np.flipud(blocked),0,254),blocked)
    return inputs


def test_recorded_wall_case_needs_heading_search_even_at_ideal_reverse_yaw(tmp_path):
    inputs=wall_near_return_endpoint(tmp_path)
    session,manifest,trial,reference,params,data=inputs
    measured=[2.5740952344,-.4350042751,-3.1166685767]
    nominal=float(experiment.wrap(-.0196300537+np.pi))
    for yaw in (measured[2],nominal):
        pose=[*measured[:2],yaw]
        _,path=experiment.reanchor_trial(manifest,trial,reference,pose)
        with pytest.raises(PathBlockedError):
            batch.check_placed_geometry(path,pose,params,data)
    before=params.read_bytes()
    target,candidates=batch.clear_turnaround_target(manifest,trial,reference,params,data,measured,nominal)
    assert target is not None
    np.testing.assert_array_equal(target[:2],measured[:2])
    assert candidates[-1]['offset_deg']==-2
    _,path=experiment.reanchor_trial(manifest,trial,reference,target)
    batch.check_placed_geometry(path,target,params,data,clearance=.03)
    assert params.read_bytes()==before
    assert experiment.arclength(path[:,:2])[-1]==pytest.approx(2.5)


def test_settled_wall_collision_causes_bounded_correction_and_fresh_placement(tmp_path):
    inputs=wall_near_return_endpoint(tmp_path)
    state=dict(pose=[2.5734670758,-.4157926223,1.5171818047])
    targets=[]
    def turn(target):
        targets.append(target.tolist())
        state['pose']=([2.5740952344,-.4350042751,-3.1166685767]
                       if len(targets)==1 else target.tolist())
    record=batch.checked_retry_turnaround(*inputs,lambda:dict(state),turn,tmp_path/'turns')
    assert len(targets)==2 and record['stopped_path_checked']
    assert 'collision' in record['attempts'][0]
    assert not record['attempts'][0]['stopped_path_checked']
    np.testing.assert_array_equal(targets[1][:2],record['attempts'][0]['after']['pose'][:2])
    session,manifest,trial,reference,params,data=inputs
    captured,placed,path=batch.checked_current_placement(manifest,trial,reference,params,data,
        lambda:record['after'],no_alignment,tmp_path/'placement',current_only=True)
    assert captured==record['after']
    batch.check_placed_geometry(path,captured['pose'],params,data,clearance=.03)
    assert not (session/'runs'/trial['id']).exists()


def test_heading_search_cannot_expand_beyond_five_degrees_or_ignore_wall(tmp_path):
    inputs=wall_near_return_endpoint(tmp_path)
    session,manifest,trial,reference,params,data=inputs
    # Wall spanning the entire end of every candidate heading.
    data[2][:,24:29]=True
    target,candidates=batch.clear_turnaround_target(manifest,trial,reference,params,data,
                                                    [2.574,-.435,1.5],np.pi)
    assert target is None and len(candidates)==11
    assert all(abs(c['offset_deg'])<=5 and not c['clear'] for c in candidates)


def test_lost_localization_after_turn_never_sends_a_correction(tmp_path):
    inputs=recorded_predecessor(tmp_path)
    calls=[]
    def capture():
        if calls: raise RuntimeError('Fresh TF required')
        return dict(pose=[2.4,0.,1.5])
    with pytest.raises(RuntimeError,match='Fresh TF required'):
        batch.checked_retry_turnaround(*inputs,capture,lambda p:calls.append(p.tolist()),tmp_path/'turns')
    assert len(calls)==1


def test_resume_after_moving_back_to_start_uses_current_heading(tmp_path):
    inputs=wall_near_return_endpoint(tmp_path)
    session,manifest,trial,path,params,data=inputs
    first_result=session/'runs'/manifest['trials'][0]['id']/'result.json'
    experiment.write_json(first_result,dict(success=True,status='succeeded',
        final_pose=[2.572894,-.4162235,1.5169065]))
    before={p:p.read_bytes() for p in (session/'runs').rglob('*') if p.is_file()}
    current=[.1277089939,-.4534532819,.0067248649]
    record=batch.checked_retry_turnaround(*inputs,lambda:dict(pose=current),no_alignment,tmp_path/'turns',
                                         resuming_first=True)
    assert record['start_mode']=='relocated_current_pose' and record['stopped_path_checked']
    assert record['attempts']==[] and record['turn_required'] is False
    assert record['distance_from_previous_stop_m']==pytest.approx(2.44547,abs=1e-5)
    assert record['after']['pose']==current
    captured,placed,actual=batch.checked_current_placement(manifest,trial,path,params,data,
        lambda:record['after'],no_alignment,tmp_path/'placement',current_only=True)
    np.testing.assert_allclose(experiment.start_pose(manifest,placed),current)
    assert actual[-1,0]>2.6
    assert experiment.arclength(actual[:,:2])[-1]==pytest.approx(2.5)
    assert all(p.read_bytes()==content for p,content in before.items())


def test_repositioned_resume_keeps_map_check(tmp_path):
    inputs=wall_near_return_endpoint(tmp_path)
    with pytest.raises(RuntimeError,match='repositioned current pose is blocked'):
        batch.checked_retry_turnaround(*inputs,lambda:dict(pose=[.1277,-.4534,np.pi]),
                                      no_alignment,tmp_path/'turns',resuming_first=True)
    record=json.loads(next((tmp_path/'turns').glob('*.json')).read_text())
    assert record['start_mode']=='relocated_current_pose'
    assert 'collision' in record and not record['stopped_path_checked']


def test_continuous_execution_still_turns_even_if_pose_changed_far(tmp_path):
    inputs=recorded_predecessor(tmp_path)
    state=dict(pose=[2.4,1.,0.]);calls=[]
    def turn(target):
        calls.append(target.tolist());state['pose']=target.tolist()
    record=batch.checked_retry_turnaround(*inputs,lambda:dict(state),turn,tmp_path/'turns')
    assert record['start_mode']=='previous_trial_return' and calls


@pytest.mark.parametrize('pose', [None,[0.,0.],[0.,0.,float('nan')]])
def test_resume_requires_valid_previous_stop_for_direction_decision(tmp_path,pose):
    inputs=recorded_predecessor(tmp_path)
    session,manifest,*_=inputs
    result=dict(success=True,status='succeeded')
    if pose is not None: result['final_pose']=pose
    # Deliberately malformed historical result, including a non-finite value.
    (session/'runs'/manifest['trials'][0]['id']/'result.json').write_text(json.dumps(result))
    with pytest.raises(ValueError,match='previous final pose is missing or invalid'):
        batch.checked_retry_turnaround(*inputs,lambda:dict(pose=[0.,0.,0.]),no_alignment,tmp_path/'turns',
                                      resuming_first=True)
