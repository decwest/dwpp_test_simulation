"""Network-isolated start capture, automatic returns, profile changes and cancel."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import rclpy
import yaml

from ros_access_controller_smoke import (Plant, spin_for, spin_until, controller_stack,
                                         synthetic_environment_path)
from ros_access_end_to_end_smoke import run_while_spinning
from access_smoke_outcomes import assert_recorded_outcome, assert_retry_bookkeeping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import dwvp_access_experiment as experiment
import dwvp_access_batch as batch


class ControllablePlant(Plant):
    def __init__(self):
        self.send_scan = True
        super().__init__()

    def publish_scan(self):
        if self.send_scan:
            super().publish_scan()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--all-methods', action='store_true')
    args = parser.parse_args()
    assert Path('/.dockerenv').exists() and set(os.listdir('/sys/class/net')) == {'lo'}
    assert os.environ.get('ROS_LOCALHOST_ONLY') == '1'
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    params = ROOT / 'params/hsrb_dwvp_access_params.yaml'
    config = experiment.default_config()
    for name, condition in config['conditions'].items():
        if not args.all_methods:
            condition['methods'] = ['DWVP']
        if name != 'E2_environment':
            condition['length_m'] = .8
            condition['evaluation']['goal_margin_m'] = .1
        if name.startswith('E1_orientation'):
            condition['orientation_start_m'] = .2
            condition['orientation_length_m'] = .15
    config['conditions']['E1_lateral']['start_pose'] = [0., .2, 0.]
    expected_trials = sum(len(c['methods']) for c in config['conditions'].values())
    configfile = out / 'synthetic_config.yaml'
    configfile.write_text(yaml.safe_dump(config))
    route = out / 'E2_environment.csv'
    # Use the same representative E2 arc as the controller/recorder checks;
    # keep the author's short E1 geometry and all workflow assertions unchanged.
    np.savetxt(route, synthetic_environment_path(),
               delimiter=',', header='x,y,yaw', comments='')
    Image.fromarray(np.full((200, 200), 254, dtype=np.uint8)).save(out / 'map.pgm')
    mapfile = out / 'map.yaml'
    mapfile.write_text(yaml.safe_dump(dict(image='map.pgm', resolution=.05,
        origin=[-5., -5., 0.], mode='trinary', negate=0, occupied_thresh=.65, free_thresh=.196)))
    rclpy.init(); plant = ControllablePlant()
    try:
        spin_for(plant, 1.)
        session = out / 'session'
        run_while_spinning(plant, [sys.executable, str(ROOT/'scripts/dwvp_access_experiment.py'),
            'prepare', '--output', str(session), '--params', str(params), '--config', str(configfile),
            '--environment-path', str(route), '--start-from-current', '--repeats', '1'] +
            (['--bidirectional'] if args.bidirectional else []), out/'capture.log', 30.)
        manifest = json.loads((session/'manifest.json').read_text())
        assert (session/'start_capture.json').is_file()
        for name in experiment.CONDITIONS[:-1]:
            np.testing.assert_allclose(manifest['starts'][name]['map_pose'], [0, 0, 0], atol=1e-5)
        command = [sys.executable, str(ROOT/'scripts/dwvp_access_batch.py'),
                   '--session', str(session), '--map', str(mapfile), '--repeats', '1']
        # Exercise the author's explicit continuation workflow for the quarter
        # VP_CLIP endpoint case. Outcome assertions below still reject failures
        # for every other case; no recorder or controller criterion is relaxed.
        if args.all_methods:
            command += ['--continue-on-endpoint-failure']
        before = plant.raw_count
        run_while_spinning(plant, command + ['--dry-run'], out/'dry_run.log', 30.)
        assert plant.raw_count == before and not (session/'runs').exists()
        run_while_spinning(plant, command, out/'batch.log', 900. if args.all_methods else 300.)
        folder = next((session/'batches').iterdir())
        status = json.loads((folder/'status.json').read_text())
        assert status['status'] == 'completed', status
        assert len(status['completed']) == expected_trials, status
        assert (len(status['alignments']) == expected_trials-1 if args.bidirectional
                else len(status['returns']) == expected_trials)
        report = experiment.summarize(session)
        assert report['recorded'] == expected_trials
        selected = [t for t in manifest['trials'] if t['repeat'] == 1]
        failures = []
        for trial in selected:
            result = assert_recorded_outcome(session, manifest, trial)
            summary = next(t for t in report['trials'] if t['trial_id'] == trial['id'])
            assert summary['success'] == result['success'], summary
            if not result['success']:
                failures.append(trial['id'])
        assert status['failed_trials'] == failures, status
        retry = assert_retry_bookkeeping(session, out/'expected_endpoint_retry')
        transfers = folder / ('alignments' if args.bidirectional else 'returns')
        assert len(list(transfers.glob('*/result.json'))) == (expected_trials-1 if args.bidirectional else expected_trials)
        for resultfile in transfers.glob('*/result.json'):
            result = json.loads(resultfile.read_text())
            assert result['purpose'] == 'reposition_only_not_an_experiment_trial' and result['success']
        if args.bidirectional:
            assert [t['direction'] for t in manifest['trials']] == ['forward' if i%2==0 else 'reverse' for i in range(expected_trials)]
            assert {t['direction'] for t in report['trials']} == {'forward', 'reverse'}
            assert report['pending'] == 0
        else:
            assert np.hypot(plant.x, plant.y) <= .1 and abs(float(experiment.wrap(plant.yaw))) <= .3
        print('PASS: measured start, all five conditions, acceleration changes, measured directions and separate positioning', flush=True)
        # A completed session is never replayed, even if the previous batch exited.
        before = plant.raw_count
        with (out/'duplicate.log').open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            spin_until(plant, lambda: process.poll() is not None, 20)
        assert process.returncode != 0 and plant.raw_count == before
        spin_for(plant, 2.); plant.reset([0, 0, 0]); spin_for(plant, .5)
        # Reproduce a batch stopped after a successful outbound trial, before
        # turning to its reverse start. Resume must align and keep that run intact.
        resume_config = experiment.default_config()
        resume_config['conditions']['E1_lateral'].update(methods=['DWVP'], length_m=.8,
                                                       start_pose=[0., .2, 0.])
        resume_file = out/'resume_config.yaml'; resume_file.write_text(yaml.safe_dump(resume_config))
        resume_session = out/'resume_session'
        resume_manifest = experiment.prepare(resume_session, params, config_file=resume_file,
            current_start=[0, 0, 0], bidirectional=True, conditions=['E1_lateral'], repeats=[1, 2])
        first, second = resume_manifest['trials']
        with controller_stack(plant, resume_session/first['params_file'], out/'resume_first_stack'):
            run_while_spinning(plant, [sys.executable, str(ROOT/'scripts/dwvp_access_experiment.py'),
                'run', '--session', str(resume_session), '--trial', first['id']], out/'resume_first.log', 50.)
        original = {p.name: p.read_bytes() for p in (resume_session/'runs'/first['id']).iterdir()}
        resume_command = [sys.executable, str(ROOT/'scripts/dwvp_access_batch.py'), '--session',
                          str(resume_session), '--map', str(mapfile), '--resume']
        before = plant.raw_count
        run_while_spinning(plant, resume_command+['--dry-run'], out/'resume_dry_run.log', 20.)
        assert plant.raw_count == before
        run_while_spinning(plant, resume_command, out/'resume_batch.log', 100.)
        resumed = json.loads(next((resume_session/'batches').glob('*/status.json')).read_text())
        assert resumed['status']=='completed' and resumed['completed']==[second['id']] and resumed['resume']
        assert resumed['alignments']==['resume_'+second['id']], resumed
        assert original == {p.name: p.read_bytes() for p in (resume_session/'runs'/first['id']).iterdir()}
        before = plant.raw_count
        run_while_spinning(plant, resume_command, out/'resume_completed.log', 20.)
        assert plant.raw_count == before and len(list((resume_session/'batches').iterdir()))==1
        print('PASS: resume aligns to reverse start, preserves the outbound record and does not replay completed trials', flush=True)
        spin_for(plant, 2.); plant.reset([0, 0, 0]); spin_for(plant, .5)
        # DWVP's terminal heading controller supports a position-preserving
        # turn; a measured path still forbids duplicate positions.
        turn_session = out/'turn_session'
        turn_manifest = experiment.prepare(turn_session, params, current_start=[0, 0, 0])
        turn_trial = turn_manifest['trials'][0]
        packet = dict(start=[0, 0, 0], path=[[0, 0, 0], [0, 0, np.pi]], output=str(out/'turn_record'))
        packet_file = out/'turn.json'; packet_file.write_text(json.dumps(packet))
        with controller_stack(plant, turn_session/turn_trial['params_file'], out/'turn_stack'):
            run_while_spinning(plant, [sys.executable, str(ROOT/'scripts/dwvp_access_batch.py'), '_transfer',
                '--session', str(turn_session), '--trial', turn_trial['id'], '--packet', str(packet_file)],
                out/'turn.log', 40.)
            spin_for(plant, 2.)
            assert np.hypot(plant.x, plant.y) < .005 and abs(float(experiment.wrap(plant.yaw-np.pi))) <= .3
        print('PASS: in-place start-heading adjustment preserves position', flush=True)
        for scenario in ('interrupt', 'missing_scan'):
            spin_for(plant, 2.)
            plant.reset([0, 0, 0]); plant.send_scan = True; spin_for(plant, .5)
            broken = out / scenario
            experiment.prepare(broken, params, config_file=configfile, current_start=[0, 0, 0],
                               bidirectional=args.bidirectional, conditions=['E1'], repeats=[1])
            command = [sys.executable, str(ROOT/'scripts/dwvp_access_batch.py'),
                       '--session', str(broken), '--map', str(mapfile), '--conditions', 'E1', '--repeats', '1']
            with (out/(scenario+'.log')).open('w') as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    spin_until(plant, lambda: np.hypot(plant.x, plant.y) > .03 or process.poll() is not None, 50)
                    assert process.poll() is None, (out/(scenario+'.log')).read_text()
                    if scenario == 'interrupt':
                        process.send_signal(signal.SIGINT)
                    else:
                        plant.send_scan = False
                    spin_until(plant, lambda: process.poll() is not None, 30)
                finally:
                    if process.poll() is None:
                        process.kill(); process.wait()
                assert process.returncode != 0
            plant.send_scan = True; spin_for(plant, 2.)
            statusfile = next((broken/'batches').glob('*/status.json'))
            failed = json.loads(statusfile.read_text())
            assert failed['status'] in ('failed', 'interrupted') and not failed['completed'], failed
            runs = list((broken/'runs').glob('*/result.json'))
            assert len(runs) == 1
            result = json.loads(runs[0].read_text())
            assert result['cancellation_confirmed'] is True and result['success'] is False, result
            assert np.hypot(plant.applied.linear.x, plant.applied.linear.y) < 1e-6
            assert abs(plant.applied.angular.z) < 1e-6
            print(f'PASS: {scenario} cancels goal, stops motion and does not start next trial', flush=True)
        (out/'report.json').write_text(json.dumps(dict(physical_trials=0, synthetic_trials=expected_trials,
            expected_endpoint_case=['E1_orientation_quarter','VP_CLIP'],
            failed_trial_ids=failures, retry_bookkeeping=retry,
            methods=sorted({t['controller'] for t in manifest['trials']}),
            bidirectional=args.bidirectional, synthetic_positioning=expected_trials-1 if args.bidirectional else expected_trials,
            resumed_reverse_trial=True, resume_preserves_success=True, completed_resume_no_goals=True,
            in_place_turn=True, current_start=True, dry_run_no_goals=True,
            duplicate_rejected=True, interruption_cancels=True, sensor_loss_cancels=True), indent=2)+'\n')
    finally:
        plant.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()
