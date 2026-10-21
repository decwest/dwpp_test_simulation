"""Actual ROS CLI with a synthetic base, strictly inside --network none Docker."""
import argparse
from collections import deque
import csv
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node

ROOT = Path(__file__).resolve().parents[1]


class Plant(Node):
    def __init__(self):
        super().__init__('omni_base_controller')
        self.declare_parameter('odometry_publish_rate', 100.)
        self.subscription = self.create_subscription(Twist, '/step_test/cmd_vel', self.command, 10)
        self.publisher = self.create_publisher(Odometry, '/step_test/odom', 10)
        self.timer = self.create_timer(.01, self.tick)
        self.reset('normal')

    def reset(self, scenario):
        self.scenario = scenario
        self.latest = 0.
        self.queue = deque()
        self.velocity = self.x = 0.
        self.first_motion = None
        self.last_tick = time.monotonic()
        self.commands = []

    def command(self, msg):
        now = time.monotonic()
        self.latest = msg.linear.x
        self.queue.append((now, self.latest))
        self.commands.append((now, self.latest))
        if self.latest and self.first_motion is None:
            self.first_motion = now

    def tick(self):
        now = time.monotonic(); dt = now-self.last_tick; self.last_tick = now
        while len(self.queue) > 1 and self.queue[1][0] <= now-.025:
            self.queue.popleft()
        target = self.queue[0][1] if self.queue and self.queue[0][0] <= now-.025 else 0.
        self.velocity += (target-self.velocity)*(1-math.exp(-dt/.06))
        self.x += dt*self.velocity
        if self.scenario == 'stale' and self.first_motion and now-self.first_motion > .12:
            return
        msg = Odometry(); msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'; msg.child_frame_id = 'base_footprint_wheel'
        msg.pose.pose.position.x = self.x; msg.pose.pose.orientation.w = 1.
        msg.twist.twist.linear.x = self.velocity
        self.publisher.publish(msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    assert Path('/.dockerenv').exists() and set(os.listdir('/sys/class/net')) == {'lo'}
    assert os.environ.get('ROS_LOCALHOST_ONLY') == '1'
    args.output.mkdir(parents=True, exist_ok=False)
    rclpy.init(); plant = Plant()
    results = {}
    child = None
    try:
        for scenario in ('normal', 'stale', 'interrupt'):
            plant.reset(scenario)
            folder = args.output/scenario
            command = [sys.executable, str(ROOT/'scripts/dwvp_actuation_step.py'),
                       '--workspace', str(args.output), '--output', str(folder), '--yes', '--no-stop-robot-joy',
                       '--axes', 'x', '--linear-speeds', '.1', '--repeats', '1', '--baseline', '.3', '--hold', '.6',
                       '--cmd-topic', '/step_test/cmd_vel', '--odom-topic', '/step_test/odom',
                       '--clock-uncertainty-ms', '0']
            with (args.output/(scenario+'.log')).open('w') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                deadline = time.monotonic()+35
                interrupted = False
                while child.poll() is None:
                    rclpy.spin_once(plant, timeout_sec=.005)
                    if scenario == 'interrupt' and plant.first_motion and not interrupted:
                        child.send_signal(signal.SIGINT); interrupted = True
                    if time.monotonic() > deadline:
                        raise RuntimeError(f'{scenario} timed out')
            state = json.loads((folder/'session.json').read_text())
            with (folder/'pulses.csv').open() as stream:
                pulses = list(csv.DictReader(stream))
            assert plant.commands and plant.commands[-1][1] == 0
            assert state['final_stop']['zero_messages'] > 0
            if scenario == 'normal':
                assert child.returncode == 0 and state['status'] == 'completed'
                assert len(pulses) == 2 and all(r['status'] == 'completed' for r in pulses)
                assert [float(r['amplitude']) for r in pulses] == [.1, -.1]
                assert state['final_stop']['stationary_verified']
                assert (folder/'base_parameters.yaml').exists()
                report = json.loads((folder/'analysis/summary.json').read_text())
                assert len(report['metrics']) == 4
                for row in report['metrics']:
                    assert .01 < row['onset_upper_s'] < .12
                    assert .8 < row['steady_gain'] < 1.2
            else:
                assert child.returncode != 0 and state['status'] != 'completed'
                assert len(pulses) == 1 and pulses[0]['status'] != 'completed'
                assert not any(value < 0 for _, value in plant.commands), 'No next pulse after failure'
                if scenario == 'stale':
                    assert not state['final_stop']['stationary_verified']
                    assert 'stale' in state['error']
            results[scenario] = dict(exit_code=child.returncode, status=state['status'],
                                     pulse_count=len(pulses), final_stop=state['final_stop'])
            print(scenario, results[scenario], flush=True)
        (args.output/'validation.json').write_text(json.dumps(dict(status='passed', scenarios=results,
            network='loopback only, Docker --network none', synthetic=True, robot_motion=False), indent=2)+'\n')
    finally:
        if child is not None and child.poll() is None:
            child.terminate(); child.wait(timeout=5)
        plant.destroy_node(); rclpy.try_shutdown()


if __name__ == '__main__':
    main()
