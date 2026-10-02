"""Run the recorder against real Humble controllers and a synthetic plant.

Requires Docker --network none, ROS_LOCALHOST_ONLY=1, and a sourced workspace
containing the DWPP and omnidirectional DWVP plugins. This is integration
validation only: no robot, AMCL, TMC node, or paper performance trial is used.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

import rclpy
from lifecycle_msgs.msg import Transition

from ros_access_controller_smoke import Plant, spin_for, spin_until, transition


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'experiment', ROOT / 'scripts/dwvp_access_experiment.py')
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)


def run_while_spinning(plant, command, output, timeout):
    with output.open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            spin_until(plant, lambda: process.poll() is not None, timeout)
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                spin_until(plant, lambda: process.poll() is not None, 8.0)
        assert process.returncode == 0, output.read_text()


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
    processes, logs = [], []
    rclpy.init()
    plant = Plant()
    try:
        for package, executable, remaps in (
            ('nav2_controller', 'controller_server', ['cmd_vel:=/cmd_vel_nav']),
            ('nav2_velocity_smoother', 'velocity_smoother', [
                'cmd_vel:=/cmd_vel_nav', 'cmd_vel_smoothed:=/omni_base_controller/cmd_vel']),
        ):
            log = (output / f'{executable}.log').open('w')
            logs.append(log)
            command = [f'/opt/ros/humble/lib/{package}/{executable}',
                       '--ros-args', '--params-file', str(params)]
            for remap in remaps:
                command.extend(['-r', remap])
            processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT))
        for name in ('controller_server', 'velocity_smoother'):
            transition(plant, name, Transition.TRANSITION_CONFIGURE)
            transition(plant, name, Transition.TRANSITION_ACTIVATE)
        snapshots = {}
        for name in ('controller_server', 'velocity_smoother'):
            snapshots[name] = output / f'{name}_runtime.yaml'
            run_while_spinning(
                plant, ['ros2', 'param', 'dump', '/' + name], snapshots[name], 20.0)
        for controller in ('RPP', 'DWPP', 'MPPI', 'DWVP'):
            experiment.verify_runtime_parameters(params, snapshots, controller)
            print(f'Actual runtime configuration verified: {controller}', flush=True)
        spin_for(plant, 0.5)
        session = output / 'session'
        experiment.prepare(session, params, [0.0, 0.0, 0.0])
        run_while_spinning(plant, [
            sys.executable, str(ROOT / 'scripts/dwvp_access_experiment.py'), 'run',
            '--session', str(session), '--trial', 'B_path1_DWVP_r1'],
            output / 'recorder.log', 60.0)
        report = experiment.summarize(session)
        trial = report['trials'][0]
        result = json.loads((session / 'runs/B_path1_DWVP_r1/result.json').read_text())
        summary = {
            'purpose': 'Synthetic integration smoke; not paper performance data',
            'verified_controller_parameters': ['RPP', 'DWPP', 'MPPI', 'DWVP'],
            'trial': trial, 'group': report['groups'][0], 'result': result,
        }
        (output / 'report.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary, indent=2), flush=True)
        assert trial['success'], trial
        assert trial['valid_pose_samples'] > 30, trial
        assert report['groups'][0]['valid_successes'] == 1, summary
        assert plant.max_vy > 0.01, 'Independent heading corner must produce lateral motion'
        print('PASS: actual runtime parameters, FollowPath, recorder, and valid summary', flush=True)
    finally:
        for process in processes:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
        for process in processes:
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in logs:
            log.close()
        plant.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
