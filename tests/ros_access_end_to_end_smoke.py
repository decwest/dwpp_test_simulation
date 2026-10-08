"""Run the recorder against real Humble controllers and a synthetic plant.

Requires Docker --network none, ROS_LOCALHOST_ONLY=1, and a sourced workspace
containing the DWPP and omnidirectional DWVP plugins. This is integration
validation only: no robot, AMCL, TMC node, or paper performance trial is used.
"""

import argparse
from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import yaml
import numpy as np

import rclpy
from lifecycle_msgs.msg import Transition

from ros_access_controller_smoke import (Plant, spin_for, spin_until, controller_stack,
                                         synthetic_environment_path, check_acceleration)
from access_smoke_outcomes import (
    EXPECTED_ENDPOINT_CASES, assert_recorded_outcome, assert_retry_bookkeeping)


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'experiment', ROOT / 'scripts/dwvp_access_experiment.py')
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)


def run_while_spinning(plant, command, output, timeout, *, separate_stderr=False):
    stderr_output = output.with_suffix('.stderr.log') if separate_stderr else None
    with output.open('w') as log, ExitStack() as stack:
        errors = stack.enter_context(stderr_output.open('w')) if stderr_output else subprocess.STDOUT
        process = subprocess.Popen(command, stdout=log, stderr=errors)
        try:
            spin_until(plant, lambda: process.poll() is not None, timeout)
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                spin_until(plant, lambda: process.poll() is not None, 8.0)
        assert process.returncode == 0, output.read_text() + (
            stderr_output.read_text() if stderr_output else '')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    assert Path('/.dockerenv').exists(), 'Use an isolated Docker container'
    assert set(os.listdir('/sys/class/net')) == {'lo'}, 'Use Docker --network none'
    assert os.environ.get('ROS_LOCALHOST_ONLY') == '1', 'Set ROS_LOCALHOST_ONLY=1'
    output = args.output or Path(tempfile.mkdtemp(prefix='access-end-to-end-'))
    output.mkdir(parents=True, exist_ok=True)
    params = ROOT / 'params/hsrb_dwvp_access_params.yaml'
    print(f'Actual controller + recorder integration artifacts: {output}', flush=True)
    session = output/'session'
    route = output/'environment.csv'
    np.savetxt(route, synthetic_environment_path(), delimiter=',', header='x,y,yaw', comments='')
    # Freeze only the repeat exercised below so failed-only retry selection can
    # validate a completed session, with no invented records for unrun repeats.
    manifest = experiment.prepare(session, params, [0.,0.,0.], route, repeats=[1])
    rclpy.init()
    plant = Plant()
    acceleration_reports = {'half': {}, 'quarter': {}}
    config = experiment.default_config()
    try:
        for profile, conditions in (
            ('nominal', ['E1_lateral', 'E1_orientation_nominal']),
            ('environment', ['E2_environment']),
            ('half', ['E1_orientation_half']),
            ('quarter', ['E1_orientation_quarter']),
        ):
            frozen = session / manifest['parameter_sets'][conditions[0]]['file']
            with controller_stack(plant, frozen, output / profile):
                snapshots = {}
                for name in ('controller_server', 'velocity_smoother'):
                    snapshots[name] = output / f'{profile}_{name}_runtime.yaml'
                    snapshots[name].write_text(yaml.safe_dump(experiment.snapshot_parameters(plant, name)))
                for controller in experiment.CONTROLLERS:
                    experiment.verify_runtime_parameters(frozen, snapshots, controller)
                if profile == 'nominal':
                    check_rejections(plant, session, output)
                trials = [t for t in manifest['trials'] if t['repeat']==1 and t['task'] in conditions]
                for selected in trials:
                    controller, condition = selected['controller'], selected['task']
                    spin_for(plant,2.3)
                    plant.reset(experiment.start_pose(manifest, selected))
                    spin_for(plant,.5)
                    trial_id=selected['id']
                    run_while_spinning(plant,[sys.executable,str(ROOT/'scripts/dwvp_access_experiment.py'),'preflight',
                        '--session',str(session),'--trial',trial_id],output/f'{trial_id}_preflight.log',40.)
                    run_while_spinning(plant,[sys.executable,str(ROOT/'scripts/dwvp_access_experiment.py'),'run',
                        '--session',str(session),'--trial',trial_id],output/f'{trial_id}_recorder.log',160.)
                    report=experiment.summarize(session)
                    trial=next(t for t in report['trials'] if t['trial_id']==trial_id)
                    result = assert_recorded_outcome(session, manifest, selected)
                    assert trial['success'] == result['success'], trial
                    assert trial['fresh_pose_samples']>30 and not trial['data_errors'], trial
                    if condition == 'E2_environment':
                        window = config['conditions'][condition]['evaluation']
                        length = experiment.arclength(synthetic_environment_path()[:, :2])[-1]
                        assert trial['evaluation_start_m'] == window['start_m'], trial
                        assert abs(trial['evaluation_end_m'] - (length - window['goal_margin_m'])) < 1.e-9, trial
                        assert trial['evaluation_complete'], trial
                    commands=report['command_diagnostics'][trial_id]
                    assert commands['constraint_total_samples']>10, commands
                    assert commands['constraint_total_samples']==commands['constraint_evaluable_samples']+commands['constraint_unknown_samples'], commands
                    assert trial['constraint_violation_pct']==commands['constraint_violation_pct'], trial
                    expected=100.*commands['constraint_violation_samples']/commands['constraint_evaluable_samples']
                    assert abs(trial['constraint_violation_pct']-expected)<1e-10, trial
                    assert trial['constraint_violation_lower_pct'] <= expected <= trial['constraint_violation_upper_pct'], trial
                    assert trial['constraint_unknown_flag'] == (trial['constraint_unknown_pct']>5.), trial
                    assert commands['pairing_clock']=='receive_monotonic_s', commands
                    for key in ('eval_max_position_error_m','eval_mean_position_error_m',
                                'eval_position_error_integral_m_s','eval_max_heading_error_deg',
                                'eval_mean_heading_error_deg','eval_heading_error_integral_deg_s'):
                        assert trial[key] is not None, trial
                    timing=report['controller_timing'][trial_id]
                    assert timing['samples']>10 and timing['sequence_gaps']==0 and timing['failed_calls']==0, timing
                    assert timing.get('sequence_nonincreasing',0)==0, timing
                    assert timing['invalid_samples']==0, timing
                    # The controller server can publish an extra terminal zero command.
                    assert abs(timing['raw_minus_successful_timing_samples'])<=3, timing
                    group=next(g for g in report['groups'] if g['task']==condition and g['controller']==controller)
                    # Quality accounting remains visible; flags do not discard observations.
                    eligible = (trial['success'] and trial['invalid_after_warmup_samples']==0
                                and trial['missing_command_prefix_s'] is not None
                                and trial['missing_command_prefix_s']<=1/manifest['control_frequency_hz'])
                    assert group['recorded']==1 and group['succeeded']==int(trial['success']), group
                    assert group['failed_or_incomplete']==int(not trial['success']), group
                    assert group['valid_successes']==int(eligible), group
                    assert group['travel_time_s_n']==int(trial['success']), group
                    assert group['constraint_violation_pct_n']==1, group
                    assert group['compute_time_mean_ms_n']==1, group
                    if profile in acceleration_reports:
                        acceleration_reports[profile][controller] = check_acceleration(plant, config, condition)
                        assert trial['acceleration_scale'] == config['conditions'][condition]['acceleration_scale']
                    print(f'Actual recorder verified: {trial_id}; quality-qualified={eligible}', flush=True)
                spin_for(plant, 2.3)
        retry = assert_retry_bookkeeping(session, output/'expected_endpoint_retry')
        (output/'report.json').write_text(json.dumps({'purpose':'Synthetic controller and recorder integration only',
            'physical_trials':0,'start_pose_rejection':True,'unassigned_rejection':True,
            'expected_endpoint_cases':EXPECTED_ENDPOINT_CASES, 'retry_bookkeeping':retry,
            'half_acceleration':acceleration_reports['half'],
            'quarter_acceleration':acceleration_reports['quarter'],'summary':report},indent=2)+'\n')
        print('PASS: all seven controllers, assigned conditions, half/quarter acceleration, timing, recorder and summary',flush=True)
    finally:
        plant.destroy_node()
        rclpy.shutdown()


def check_rejections(plant, session, output):
    for trial_id, reason in [('E1_lateral_DWVP_r1', 'frozen condition start pose'),
                             ('E1_lateral_MPPI_r1', 'unassigned'),
                             ('E1_orientation_half_DWVP_r1', 'Runtime parameter mismatch'),
                             ('E1_orientation_quarter_DWVP_r1', 'Runtime parameter mismatch'),
                             ('E1_orientation_quarter_DWB_r1', 'unassigned')]:
        rejected = output / f'rejected_{trial_id}.log'
        try:
            run_while_spinning(plant,[sys.executable,str(ROOT/'scripts/dwvp_access_experiment.py'),'preflight',
                '--session',str(session),'--trial',trial_id],rejected,40.)
        except AssertionError:
            assert reason in rejected.read_text(), rejected.read_text()
        else:
            raise AssertionError('Invalid start or method assignment accepted')
    assert not (session/'runs').exists()


if __name__ == '__main__':
    main()
