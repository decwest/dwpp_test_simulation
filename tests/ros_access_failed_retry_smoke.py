"""Retry at current poses, turning back between legs, including across resume."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import rclpy
import yaml

from ros_access_controller_smoke import Plant, spin_for
from ros_access_end_to_end_smoke import run_while_spinning

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import dwvp_access_experiment as experiment
from dwvp_access_batch import prepare_failed_retry
from dwvp_access_path import PathBlockedError, load_map
from dwvp_access_batch import check_placed_geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    scenario=parser.add_mutually_exclusive_group()
    scenario.add_argument('--inject-turn-drift', action='store_true', help='Inject a localization shift after the first turn stops')
    scenario.add_argument('--reposition-before-resume', action='store_true', help='Move the synthetic robot back to its start before resuming')
    args = parser.parse_args()
    assert Path('/.dockerenv').exists() and set(os.listdir('/sys/class/net')) == {'lo'}
    assert os.environ.get('ROS_LOCALHOST_ONLY') == '1'
    out = args.output; out.mkdir(parents=True, exist_ok=False)
    config = experiment.default_config()
    condition = config['conditions']['E1_orientation_quarter']
    condition.update(methods=['DWVP'], length_m=.8, orientation_start_m=.2, orientation_length_m=.15)
    condition['evaluation']['goal_margin_m'] = .1
    settings = out/'config.yaml'; settings.write_text(yaml.safe_dump(config))
    Image.fromarray(np.full((200,200),254,dtype=np.uint8)).save(out/'map.pgm')
    map_file = out/'map.yaml'
    map_file.write_text(yaml.safe_dump(dict(image='map.pgm', resolution=.05, origin=[-5.,-5.,0.],
                                          mode='trinary', negate=0, occupied_thresh=.65, free_thresh=.196)))
    source = out/'original_session'
    original = experiment.prepare(source, ROOT/'params/hsrb_dwvp_access_params.yaml',
        current_start=[6,6,1.2], config_file=settings, bidirectional=True,
        conditions=['E1_orientation_quarter'], repeats=[1,2,3,4])
    # Fabricated past results test selection/preservation; only the two retries
    # below use live synthetic odometry and the actual controller/smoother stack.
    for trial in original['trials']:
        folder = source/'runs'/trial['id']; folder.mkdir(parents=True)
        success = trial['direction'] == 'forward'
        result = dict(status='succeeded' if success else 'failed', success=success)
        if not success:
            result.update(action_status=4, endpoint_policy='fixed_tolerance',
                final_pose_within_tolerances=False, final_pose=[0.,0.,.52], final_pose_fresh=True,
                final_position_error_m=.08, final_yaw_error_rad=.52, settling=dict(verified=True),
                failure_reason='endpoint_tolerance_exceeded',
                error='Final stopped pose outside tolerances or action unsuccessful: synthetic source record')
        experiment.write_json(folder/'result.json', result)
        experiment.write_json(folder/'trial.json', dict(trial=trial,
            manifest_sha256=experiment.digest(source/'manifest.json'), params_sha256=trial['params_sha256']))
    (source/'.batch.lock').touch()
    preserved = {str(p): p.read_bytes() for p in source.rglob('*') if p.is_file()}
    session = out/'retry_session'
    manifest = prepare_failed_retry(source, session)
    assert [t['direction'] for t in manifest['trials']] == ['reverse','reverse']
    first = manifest['trials'][0]
    try:
        check_placed_geometry(experiment.load_trial(session,first['id'])[2],
                              experiment.start_pose(manifest,first), session/first['params_file'], load_map(map_file))
    except PathBlockedError:
        pass  # Old starts are deliberately outside the map; retries must ignore them.
    else:
        raise AssertionError('Historical path must be outside this test map')
    command = [sys.executable, str(ROOT/'scripts/dwvp_access_batch.py'), '--session', str(session),
               '--map', str(map_file), '--start-from-current', '--continue-on-endpoint-failure']
    rclpy.init(); plant = Plant()
    injected=[]
    def inject_drift():
        if not args.inject_turn_drift or injected:
            return
        for result_file in (session/'batches').glob('*/turnarounds/*attempt_01/result.json'):
            try:
                result=json.loads(result_file.read_text())
            except (OSError, ValueError):
                continue  # The child may still be writing the result.
            assert result['success'] and result['settling']['verified']
            assert abs(plant.applied.angular.z)<1e-6
            plant.yaw += .07  # Simulate stopped localization change, not a velocity command.
            injected.append(dict(yaw_shift_rad=.07,after=str(result_file)))
    plant.create_timer(.02,inject_drift)
    try:
        plant.reset([0.,0.,.2]); spin_for(plant, 1.)
        run_while_spinning(plant, command+['--repeats',str(first['repeat'])], out/'retry_first.log', 160.)
        if args.reposition_before_resume:
            plant.reset([0.,0.,.2]); spin_for(plant, 1.)
        run_while_spinning(plant, command+['--resume'], out/'retry_resume.log', 200.)
        assert all(Path(p).read_bytes() == contents for p, contents in preserved.items())
        assert set(p.name for p in (session/'runs').iterdir()) == {t['id'] for t in manifest['trials']}
        previous_stop = [0.,0.,.2]
        previous_axis = None
        for index,trial in enumerate(manifest['trials']):
            folder = session/'runs'/trial['id']
            result = json.loads((folder/'result.json').read_text())
            assert result['success'] and result['settling']['verified'], result
            captured = json.loads((folder/'start_capture.json').read_text())
            assert captured['alignment_target'] is None
            assert captured['start_policy'] == result['start_policy'] == 'current_pose_with_turnaround'
            assert result['source_direction'] == 'reverse'
            assert result['goal_checker_id']=='general_goal_checker'
            expected_xy=[0.,0.] if args.reposition_before_resume else previous_stop[:2]
            np.testing.assert_allclose(captured['capture']['pose'][:2],expected_xy,atol=.01,rtol=0)
            actual=np.loadtxt(folder/'reference.csv',delimiter=',',skiprows=1)
            axis=actual[-1,:2]-actual[0,:2]
            if index==0:
                assert captured['turnaround'] is None
                assert abs(experiment.wrap(captured['capture']['pose'][2]-.2))<.01
            else:
                assert captured['turnaround']['stopped_heading_verified']
                direction_cos=np.dot(axis,previous_axis)/(np.linalg.norm(axis)*np.linalg.norm(previous_axis))
                if args.reposition_before_resume:
                    assert captured['turnaround']['start_mode']=='relocated_current_pose'
                    assert captured['turnaround']['stopped_path_checked']
                    assert captured['turnaround']['attempts']==[]
                    assert direction_cos>.998
                    assert abs(experiment.wrap(captured['capture']['pose'][2]-.2))<.01
                else:
                    assert captured['turnaround']['start_mode']=='previous_trial_return'
                    assert direction_cos<-.998
                    assert abs(experiment.wrap(captured['capture']['pose'][2]-(previous_stop[2]+np.pi)))>1.
            previous_axis=axis
            previous_stop = result['final_pose']
            assert (session/trial['params_file']).read_bytes() == (source/trial['params_file']).read_bytes()
        turns=[]
        for batch_folder in (session/'batches').iterdir():
            status=json.loads((batch_folder/'status.json').read_text())
            assert status['status']=='completed' and not status['alignments'] and not status['returns']
            turns.extend((batch_folder/'turnarounds').glob('*/result.json'))
        assert len(turns)==(0 if args.reposition_before_resume else 2 if args.inject_turn_drift else 1)
        assert len(injected)==int(args.inject_turn_drift)
        for result_file in turns:
            turn_result=json.loads(result_file.read_text())
            assert turn_result['success'] and turn_result['goal_checker_id']==experiment.TURNAROUND_GOAL_CHECKER
            commands=np.genfromtxt(result_file.parent/'applied.csv',delimiter=',',names=True)
            assert np.max(np.abs(commands['vx']))<1e-6 and np.max(np.abs(commands['vy']))<1e-6
            assert np.max(np.abs(commands['omega']))>.01
        report = experiment.summarize(session)
        assert report['recorded'] == 2 and report['planned'] == 2 and report['pending'] == 0
        count = plant.raw_count
        run_while_spinning(plant, command+['--resume'], out/'completed_resume.log', 20.)
        assert count == plant.raw_count
        (out/'report.json').write_text(json.dumps(dict(physical_trials=0, synthetic_retry_trials=2,
            successes=2, source_directions=['reverse','reverse'], original_records_unchanged=True,
            unscored_alignments=0, in_place_turnarounds=len(turns), injected_stopped_drift=injected,
            reverse_travel_on_resume=not args.reposition_before_resume,
            repositioned_resume_uses_current_heading=args.reposition_before_resume,
            scored_goal_checker_unchanged=True, current_pose_each_trial=True, old_starts_outside_map_ignored=True,
            successful_source_trials_not_repeated=True), indent=2)+'\n')
        print('PASS: retry resume selects current heading after repositioning, otherwise turns back; conditions and records preserved.')
    finally:
        plant.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()
