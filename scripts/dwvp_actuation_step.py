#!/usr/bin/env python3
"""Bounded body-axis step inputs and wheel-odometry response recording."""
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import signal
import shlex
import shutil
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import uuid

from dwvp_actuation_metrics import AXES, check_plotting, summarize

CMD_TOPIC = '/omni_base_controller/cmd_vel'
ODOM_TOPIC = '/omni_base_controller/wheel_odom'
PULSE_FIELDS = ['trial_id', 'axis', 'amplitude', 'repeat', 'status', 'error',
                'window_start_mono_s', 'on_mono_s', 'on_publish_end_mono_s', 'on_ros_s',
                'off_mono_s', 'off_ros_s', 'window_end_mono_s']


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, default=Path.cwd())
    p.add_argument('--output', type=Path)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true', help='Print schedule only; no ROS, SSH or output directory')
    mode.add_argument('--analyze', type=Path, help='Recalculate a recording offline; no ROS or commands')
    p.add_argument('--axes', nargs='+', choices=list(AXES), default=list(AXES))
    p.add_argument('--linear-speeds', nargs='+', type=float, default=[.05, .1, .2])
    p.add_argument('--angular-speeds', nargs='+', type=float, default=[.1, .3, .6])
    p.add_argument('--linear-limit', type=float, default=.22, help='Explicit command ceiling [m/s], not a measured maximum')
    p.add_argument('--angular-limit', type=float, default=.6, help='Explicit command ceiling [rad/s]')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--hold', type=float, default=2., help='Nonzero step duration [s]')
    p.add_argument('--baseline', type=float, default=1., help='Zero command recording before each step [s]')
    p.add_argument('--frequency', type=float, default=100., help='Command refresh rate [Hz]; does not change odometry rate')
    p.add_argument('--settle-timeout', type=float, default=8.)
    p.add_argument('--max-translation', type=float, default=1., help='Stop at this odometric distance from session start [m]')
    p.add_argument('--max-rotation', type=float, default=2.5, help='Stop at this unwrapped yaw displacement from start [rad]')
    p.add_argument('--clock-uncertainty-ms', type=float,
                   help='Justified bound on relative PC/robot clock error; enables separate source-time analysis')
    p.add_argument('--acceleration-window', type=float, default=.1, help='Local linear-fit window [s]')
    p.add_argument('--cmd-topic', default=CMD_TOPIC)
    p.add_argument('--odom-topic', default=ODOM_TOPIC)
    p.add_argument('--odom-child-frame', default='base_footprint_wheel')
    p.add_argument('--base-node', default='omni_base_controller', help='Read-only parameter snapshot target')
    p.add_argument('--yes', action='store_true', help='Explicitly skip the interactive start prompt')
    p.add_argument('--no-stop-robot-joy', action='store_true')
    p.add_argument('--robot-host', default=os.environ.get('HSR_IP', '192.168.50.10'))
    p.add_argument('--robot-user', default=os.environ.get('HSR_SSH_USER', 'administrator'))
    p.add_argument('--robot-container', default=os.environ.get('HSR_ROBOT_CONTAINER', 'docker.humble.robot.service'))
    return p


def schedule(args):
    positive = ('linear_limit', 'angular_limit', 'hold', 'baseline', 'frequency', 'settle_timeout',
                'max_translation', 'max_rotation', 'acceleration_window')
    if any(not math.isfinite(getattr(args, k)) or getattr(args, k) <= 0 for k in positive):
        raise ValueError('Durations, limits and frequency must be positive and finite')
    if args.repeats < 1 or args.frequency > 250 or args.baseline < .25:
        raise ValueError('Require repeats >= 1, frequency <= 250 Hz, baseline >= 0.25 s')
    if args.clock_uncertainty_ms is not None and (not math.isfinite(args.clock_uncertainty_ms) or args.clock_uncertainty_ms < 0):
        raise ValueError('Clock uncertainty must be finite and nonnegative')
    if not args.cmd_topic.startswith('/') or not args.odom_topic.startswith('/') or not args.odom_child_frame:
        raise ValueError('Use absolute topic names and an explicit body-fixed odometry child frame')
    trials = []
    for axis in dict.fromkeys(args.axes):
        values = args.angular_speeds if axis == 'yaw' else args.linear_speeds
        limit = args.angular_limit if axis == 'yaw' else args.linear_limit
        travel = args.max_rotation if axis == 'yaw' else args.max_translation
        if any(not math.isfinite(v) or not 0 < v <= limit for v in values):
            raise ValueError(f'{axis}: positive step levels must be <= explicitly selected ceiling {limit}')
        if max(values)*args.hold >= .8*travel:
            raise ValueError(f'{axis}: commanded step travel needs room below the displacement bound; adjust hold/levels/area')
        for level in sorted(set(values)):
            for repeat in range(1, args.repeats+1):
                for sign in (1, -1):
                    trials.append(dict(trial_id=f'{axis}_{level:g}_{"pos" if sign > 0 else "neg"}_r{repeat}',
                                       axis=axis, amplitude=sign*level, repeat=repeat))
    return trials


class Recorder:
    def __init__(self, args, folder):
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
        self.args, self.folder, self.Twist, self.rclpy = args, folder, Twist, rclpy
        self.node = Node('dwvp_actuation_step')
        if self.node.get_parameter('use_sim_time').value:
            raise RuntimeError('Physical response measurement requires wall time, not use_sim_time')
        self.cancelled = False
        self.latest = None
        self.origin = None
        self.yaw_total = 0.
        self.last_yaw = None
        self.odom_problem = None
        self.last_source = None
        self.last_command = None
        self.command_value = (0., 0., 0.)
        self.next_owner_check = 0.
        self.streams = []
        self.commands = self.writer('commands.csv', ['publish_begin_mono_s', 'publish_end_mono_s', 'publish_ros_s', 'vx', 'vy', 'omega'])
        self.odometry = self.writer('odom.csv', ['receive_mono_s', 'receive_ros_s', 'source_ros_s',
                                               'frame', 'child_frame', 'x', 'y', 'yaw', 'vx', 'vy', 'omega'])
        self.pulses = self.writer('pulses.csv', PULSE_FIELDS)
        self.subscription = self.node.create_subscription(Odometry, args.odom_topic, self.receive, qos_profile_sensor_data)
        self.publisher = self.node.create_publisher(Twist, args.cmd_topic, 10)

    def writer(self, name, fields):
        stream = (self.folder/name).open('w', newline='')
        self.streams.append(stream)
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        return writer

    def receive(self, msg):
        mono = time.monotonic()
        ros = self.node.get_clock().now().nanoseconds/1e9
        source = msg.header.stamp.sec+msg.header.stamp.nanosec/1e9
        p, q, v = msg.pose.pose.position, msg.pose.pose.orientation, msg.twist.twist
        yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
        values = [p.x, p.y, yaw, v.linear.x, v.linear.y, v.angular.z]
        self.odometry.writerow(dict(zip(['receive_mono_s', 'receive_ros_s', 'source_ros_s',
                                        'frame', 'child_frame', 'x', 'y', 'yaw', 'vx', 'vy', 'omega'],
                                       [mono, ros, source, msg.header.frame_id, msg.child_frame_id, *values])))
        if not all(math.isfinite(x) for x in values) or source <= 0:
            self.odom_problem = 'Non-finite odometry or missing source stamp'
        elif msg.child_frame_id != self.args.odom_child_frame:
            self.odom_problem = f'Unexpected odometry child frame: {msg.child_frame_id}'
        elif self.last_source is not None and source <= self.last_source:
            self.odom_problem = 'Odometry source stamps are not increasing'
        if self.last_yaw is not None:
            self.yaw_total += math.atan2(math.sin(yaw-self.last_yaw), math.cos(yaw-self.last_yaw))
        self.last_yaw, self.last_source = yaw, source
        self.latest = dict(mono=mono, source=source, ros=ros, x=p.x, y=p.y,
                           vx=v.linear.x, vy=v.linear.y, omega=v.angular.z)

    def check(self):
        if self.cancelled:
            raise KeyboardInterrupt('Operator interrupted step measurement')
        if self.odom_problem:
            raise RuntimeError(self.odom_problem)
        now = time.monotonic()
        if self.latest is None or now-self.latest['mono'] > .2:
            raise RuntimeError('Wheel odometry reception stopped or is stale')
        age = self.node.get_clock().now().nanoseconds/1e9-self.latest['source']
        if not -.05 <= age <= .5:
            raise RuntimeError(f'Odometry source clock is stale/offset ({age:.3f} s); check time synchronization')
        if self.origin is not None:
            distance = math.hypot(self.latest['x']-self.origin[0], self.latest['y']-self.origin[1])
            if distance >= self.args.max_translation or abs(self.yaw_total-self.origin[2]) >= self.args.max_rotation:
                raise RuntimeError('Selected displacement bound reached; stopping the sequence')
        if now >= self.next_owner_check:
            own = self.node.get_fully_qualified_name()
            owners = [p.node_namespace.rstrip('/')+'/'+p.node_name
                      for p in self.node.get_publishers_info_by_topic(self.args.cmd_topic)]
            if owners != [own]:
                raise RuntimeError(f'Competing velocity publishers: {owners}; expected only {own}')
            if self.publisher.get_subscription_count() == 0:
                raise RuntimeError('No base controller subscribes to the command topic')
            self.next_owner_check = now+.1

    def send(self, velocity):
        msg = self.Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = map(float, velocity)
        mono = time.monotonic()
        ros = self.node.get_clock().now().nanoseconds/1e9
        self.publisher.publish(msg)
        end = time.monotonic()
        self.commands.writerow(dict(zip(['publish_begin_mono_s', 'publish_end_mono_s', 'publish_ros_s', 'vx', 'vy', 'omega'],
                                        [mono, end, ros, *velocity])))
        self.last_command, self.command_value = mono, tuple(velocity)
        return mono, ros, end

    def poll(self, velocity=None):
        self.rclpy.spin_once(self.node, timeout_sec=.001)
        self.check()
        now = time.monotonic()
        if self.last_command is not None and any(self.command_value) and now-self.last_command > .15:
            raise RuntimeError('Command refresh missed its deadline')
        if velocity is not None and (self.last_command is None or now-self.last_command >= 1/self.args.frequency):
            self.send(velocity)

    def hold(self, velocity, duration):
        end = time.monotonic()+duration
        while time.monotonic() < end:
            self.poll(velocity)

    def stationary(self, *, send_zero=True, minimum=.5):
        deadline, stable, start = time.monotonic()+self.args.settle_timeout, None, time.monotonic()
        while time.monotonic() < deadline:
            self.poll((0., 0., 0.) if send_zero else None)
            stopped = math.hypot(self.latest['vx'], self.latest['vy']) <= .005 and abs(self.latest['omega']) <= .01
            if not stopped:
                stable = None
            elif stable is None:
                stable = time.monotonic()
            elif time.monotonic()-stable >= .5 and time.monotonic()-start >= minimum:
                return
        raise RuntimeError('Stationary wheel odometry not confirmed; no next step sent')

    def run(self, trials):
        # Read-only discovery before the first zero command.
        end = time.monotonic()+2.
        while time.monotonic() < end:
            self.rclpy.spin_once(self.node, timeout_sec=.01)
        self.stationary(send_zero=False)
        self.origin = (self.latest['x'], self.latest['y'], self.yaw_total)
        for i, trial in enumerate(trials, 1):
            row = dict.fromkeys(PULSE_FIELDS, '')
            row.update(trial, status='incomplete', window_start_mono_s=time.monotonic())
            try:
                self.hold((0., 0., 0.), self.args.baseline)
                self.stationary()
                print(f'[{i}/{len(trials)}] {trial["trial_id"]}: {trial["amplitude"]:+g}', flush=True)
                value = [0., 0., 0.]
                value[list(AXES).index(trial['axis'])] = trial['amplitude']
                self.check()
                row['on_mono_s'], row['on_ros_s'], row['on_publish_end_mono_s'] = self.send(value)
                self.hold(value, self.args.hold)
                row['off_mono_s'], row['off_ros_s'], _ = self.send((0., 0., 0.))
                self.stationary(minimum=.8)
                row['status'] = 'completed'
            except BaseException as exc:
                row.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc))
                raise
            finally:
                row['window_end_mono_s'] = time.monotonic()
                self.pulses.writerow(row)
                for stream in self.streams:
                    stream.flush()

    def close(self):
        # SIGINT/SIGTERM only set a flag, so the publisher remains alive here.
        end, stable, confirmed = time.monotonic()+2., None, False
        zero_messages = 0
        error = None
        try:
            self.send((0., 0., 0.)); zero_messages += 1
            while time.monotonic() < end:
                if time.monotonic()-self.last_command >= 1/self.args.frequency:
                    self.send((0., 0., 0.)); zero_messages += 1
                self.rclpy.spin_once(self.node, timeout_sec=.01)
                fresh = self.latest is not None and time.monotonic()-self.latest['mono'] < .2
                stopped = fresh and math.hypot(self.latest['vx'], self.latest['vy']) <= .005 and abs(self.latest['omega']) <= .01
                stable = (stable or time.monotonic()) if stopped else None
                if stable is not None and time.monotonic()-stable >= .5:
                    confirmed = True
                    break
        except Exception as exc:
            error = str(exc)
        finally:
            for stream in self.streams:
                stream.close()
            self.node.destroy_node()
        return dict(zero_messages=zero_messages, stationary_verified=confirmed, error=error)


def analyze_recording(folder, args):
    try:
        summarize(folder, clock_uncertainty_ms=args.clock_uncertainty_ms,
                  acceleration_window=args.acceleration_window)
    except Exception as exc:
        command = ['./dwvp_access.sh', 'actuation-step', '--analyze', str(folder),
                   '--acceleration-window', str(args.acceleration_window)]
        if args.clock_uncertainty_ms is not None:
            command += ['--clock-uncertainty-ms', str(args.clock_uncertainty_ms)]
        raise RuntimeError(f'Analysis failed: {exc}\nRaw recording retained at {folder}. '
                           f'Acquisition status is in session.json.\n'
                           f'Retry analysis without robot motion: {shlex.join(command)}') from exc
    print(f'Analysis: {folder}/analysis', flush=True)


def main(argv=None):
    args = parser().parse_args(argv)
    args.workspace = args.workspace.resolve()
    if args.analyze:
        folder = args.analyze if args.analyze.is_absolute() else args.workspace/args.analyze
        analyze_recording(folder, args)
        return
    trials = schedule(args)
    print(f'{len(trials)} pulses; axes={args.axes}; hold={args.hold:g} s; command refresh={args.frequency:g} Hz')
    print('Each positive pulse is followed by a verified stop, negative pulse, and verified stop.')
    if args.dry_run:
        for trial in trials:
            print(trial['trial_id'], trial['amplitude'])
        print('No ROS nodes, SSH, velocity commands or recording directories created.')
        return
    if not args.yes and not sys.stdin.isatty():
        raise ValueError('Use an interactive terminal for Enter, or explicitly select --yes')
    try:
        check_plotting()
    except Exception as exc:
        raise RuntimeError(f'Plotting dependencies unavailable: {exc}. Measurement was not started. '
                           f'Check NumPy/Matplotlib in {sys.executable}.') from exc
    from dwvp_access_workflow import require_idle
    # Same selective robot JOY handoff as the existing operator workflow.
    with (args.workspace/'.dwvp_actuation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        handoff = require_idle(args)
        print(f'Direct commands to {args.cmd_topic}; Nav2 velocity_smoother is bypassed.\n'
              f'Monitor the clear area with the emergency stop in hand. Bounds: {args.max_translation:g} m, {args.max_rotation:g} rad.', flush=True)
        if not args.yes:
            input('本体を停止し、周囲を確認してください。Enterでステップ応答の計測を開始します。')
        import rclpy
        from rclpy.signals import SignalHandlerOptions
        name = 'step_'+datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
        folder = args.output or Path('results/dwvp_actuation')/name
        folder = folder if folder.is_absolute() else args.workspace/folder
        folder.mkdir(parents=True, exist_ok=False)
        settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        state = dict(status='running', settings=settings, trials=trials, robot_teleop_handoff=handoff,
                     measurement='direct PC cmd_vel to wheel encoder odometry; onboard filters and limits retained')
        (folder/'tools').mkdir()
        state['software'] = dict(python=platform.python_version(), source_sha256={})
        for filename in ('dwvp_actuation_step.py', 'dwvp_actuation_metrics.py'):
            source = Path(__file__).resolve().parent/filename
            shutil.copy2(source, folder/'tools'/filename)
            state['software']['source_sha256'][filename] = hashlib.sha256(source.read_bytes()).hexdigest()
        write_json(folder/'session.json', state)
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        recorder = None
        old_signals = {}
        failure = None
        try:
            recorder = Recorder(args, folder)
            for sig in (signal.SIGINT, signal.SIGTERM):
                old_signals[sig] = signal.signal(sig, lambda *_: setattr(recorder, 'cancelled', True))
            from dwvp_access_experiment import snapshot_parameters
            import yaml
            try:
                base_parameters = snapshot_parameters(recorder.node, args.base_node.strip('/'), timeout_s=3.)
                (folder/'base_parameters.yaml').write_text(yaml.safe_dump(base_parameters))
            except RuntimeError as exc:
                state['parameter_snapshot_error'] = str(exc)
                print(f'Base parameter snapshot unavailable: {exc}', flush=True)
            recorder.run(trials)
            state['status'] = 'completed'
        except BaseException as exc:
            failure = exc
            state.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc))
        finally:
            if recorder is not None:
                state['final_stop'] = recorder.close()
                if not state['final_stop']['stationary_verified'] and failure is None:
                    failure = RuntimeError('Final stop not verified')
                    state.update(status='failed', error=str(failure))
            rclpy.try_shutdown()
            for sig, handler in old_signals.items():
                signal.signal(sig, handler)
            write_json(folder/'session.json', state)
        print(f'Recording: {folder} (acquisition: {state["status"]})', flush=True)
        analysis_failure = None
        if (folder/'pulses.csv').exists():
            try:
                analyze_recording(folder, args)
            except Exception as exc:
                analysis_failure = exc
        if failure is not None:
            if analysis_failure is not None:
                print(str(analysis_failure), file=sys.stderr)
            raise RuntimeError(f'Measurement stopped: {failure}; raw data retained at {folder}') from failure
        if analysis_failure is not None:
            raise RuntimeError(f'Measurement completed; post-processing failed.\n{analysis_failure}') from analysis_failure


if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
