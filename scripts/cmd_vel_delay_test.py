#!/usr/bin/env python3
"""Apply cmd_vel test signals and measure odom response delay."""

import argparse
import csv
import datetime
import math
import os
import sys
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')

from geometry_msgs.msg import Twist  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

from nav_msgs.msg import Odometry  # noqa: E402

import numpy as np  # noqa: E402

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import qos_profile_sensor_data  # noqa: E402
from rclpy.utilities import remove_ros_args  # noqa: E402


DEFAULT_OUTPUT_DIR = Path('/home/ubuntu/ros2_ws/src/dwpp_test_simulation')


def _estimate_delay(t, reference, response, max_lag_sec=2.0, min_points=8):
    finite_time = np.isfinite(t)
    if finite_time.sum() < min_points:
        return np.nan, np.nan

    dt = np.nanmedian(np.diff(t[finite_time]))
    if not np.isfinite(dt) or dt <= 0.0:
        return np.nan, np.nan

    max_lag = int(max_lag_sec / dt)
    max_lag = max(1, min(max_lag, len(t) - 1))

    best_corr = -np.inf
    best_lag = 0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            ref = reference[: len(reference) - lag]
            res = response[lag:]
        else:
            ref = reference[-lag:]
            res = response[: len(response) + lag]

        valid = np.isfinite(ref) & np.isfinite(res)
        if valid.sum() < min_points:
            continue

        ref = ref[valid] - np.nanmean(ref[valid])
        res = res[valid] - np.nanmean(res[valid])
        ref_std = np.nanstd(ref)
        res_std = np.nanstd(res)
        if ref_std < 1e-9 or res_std < 1e-9:
            continue

        corr = float(np.nanmean((ref / ref_std) * (res / res_std)))
        if corr > best_corr:
            best_corr = corr
            best_lag = lag

    if not np.isfinite(best_corr):
        return np.nan, np.nan
    return best_lag * dt, best_corr


def _threshold_delay(t, command, response, amplitude):
    if abs(amplitude) < 1e-9:
        return np.nan

    sign = 1.0 if amplitude > 0.0 else -1.0
    threshold = 0.5 * abs(amplitude)
    cmd_signal = sign * command
    res_signal = sign * response

    cmd_candidates = np.flatnonzero(cmd_signal >= threshold)
    if len(cmd_candidates) == 0:
        return np.nan

    cmd_index = int(cmd_candidates[0])
    res_candidates = np.flatnonzero(
        (np.arange(len(res_signal)) >= cmd_index) & (res_signal >= threshold)
    )
    if len(res_candidates) == 0:
        return np.nan

    return t[int(res_candidates[0])] - t[cmd_index]


def _plot_pair(t, command, measured, ylabel, output_path):
    plt.figure(figsize=(10, 5))
    plt.plot(t, command, label='command', color='tab:red', linewidth=1.8)
    plt.plot(t, measured, label='odom', color='tab:blue', linewidth=1.8)
    plt.xlabel('Time [s]')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _write_csv(output_path, rows):
    fieldnames = ['t', 'v_cmd', 'w_cmd', 'v_odom', 'w_odom']
    with output_path.open('w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(output_path, signals, args):
    t = signals['t']
    v_delay, v_corr = _estimate_delay(
        t,
        signals['v_cmd'],
        signals['v_odom'],
        max_lag_sec=args.max_lag_sec,
    )
    w_delay, w_corr = _estimate_delay(
        t,
        signals['w_cmd'],
        signals['w_odom'],
        max_lag_sec=args.max_lag_sec,
    )

    lines = [
        'cmd_vel delay test summary',
        'Positive delay means odom lags the command.',
        f'mode: {args.mode}',
        f'cmd_topic: {args.cmd_topic}',
        f'odom_topic: {args.odom_topic}',
        f'publish_rate: {args.rate:.3f} Hz',
        f'max_lag_search: {args.max_lag_sec:.3f} s',
        '',
        f'v cross-correlation delay: {v_delay:.6f} s, corr={v_corr:.6f}',
        f'w cross-correlation delay: {w_delay:.6f} s, corr={w_corr:.6f}',
    ]

    if args.mode == 'step':
        v_step_delay = _threshold_delay(
            t,
            signals['v_cmd'],
            signals['v_odom'],
            args.linear_amplitude,
        )
        w_step_delay = _threshold_delay(
            t,
            signals['w_cmd'],
            signals['w_odom'],
            args.angular_amplitude,
        )
        lines.extend(
            [
                '',
                '50% step-response delay:',
                f'v: {v_step_delay:.6f} s',
                f'w: {w_step_delay:.6f} s',
            ]
        )

    output_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _save_outputs(output_dir, prefix, rows, args):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f'{prefix}.csv'
    velocity_plot_path = output_dir / f'{prefix}_velocity_profile.png'
    angular_plot_path = output_dir / f'{prefix}_angular_velocity_profile.png'
    summary_path = output_dir / f'{prefix}_delay_summary.txt'

    _write_csv(csv_path, rows)

    signals = {
        't': np.array([row['t'] for row in rows], dtype=float),
        'v_cmd': np.array([row['v_cmd'] for row in rows], dtype=float),
        'w_cmd': np.array([row['w_cmd'] for row in rows], dtype=float),
        'v_odom': np.array([row['v_odom'] for row in rows], dtype=float),
        'w_odom': np.array([row['w_odom'] for row in rows], dtype=float),
    }
    _plot_pair(
        signals['t'],
        signals['v_cmd'],
        signals['v_odom'],
        'Linear Velocity [m/s]',
        velocity_plot_path,
    )
    _plot_pair(
        signals['t'],
        signals['w_cmd'],
        signals['w_odom'],
        'Angular Velocity [rad/s]',
        angular_plot_path,
    )
    _write_summary(summary_path, signals, args)

    return csv_path, velocity_plot_path, angular_plot_path, summary_path


class CmdVelDelayTest(Node):
    """ROS node that publishes cmd_vel test signals and records odom."""

    def __init__(self, args):
        """Initialize publishers, subscribers, and the sampling timer."""
        super().__init__('cmd_vel_delay_test')
        self.args = args
        self.created_time = self.get_clock().now().nanoseconds * 1e-9
        self.start_time = None
        self.last_odom = None
        self.done = False
        self.rows = []

        self.cmd_pub = self.create_publisher(Twist, args.cmd_topic, 10)
        self.odom_sub = self.create_subscription(
            Odometry,
            args.odom_topic,
            self._odom_callback,
            qos_profile_sensor_data,
        )
        self.timer = self.create_timer(1.0 / args.rate, self._timer_callback)

    def _odom_callback(self, msg):
        self.last_odom = msg

    def _command_at(self, t):
        if self.args.mode == 'step':
            return self._step_command_at(t)
        return self._sine_command_at(t)

    def _step_command_at(self, t):
        z = self.args.zero_duration
        d = self.args.step_duration

        if t < z:
            return 0.0, 0.0
        if t < z + d:
            return self.args.linear_amplitude, 0.0
        if t < 2.0 * z + d:
            return 0.0, 0.0
        if t < 2.0 * z + 2.0 * d:
            return 0.0, self.args.angular_amplitude
        return 0.0, 0.0

    def _sine_command_at(self, t):
        z = self.args.zero_duration
        d = self.args.sine_duration
        local_t = t - z

        if local_t < 0.0:
            return 0.0, 0.0
        if local_t < d:
            v_cmd = self.args.linear_amplitude * math.sin(
                2.0 * math.pi * self.args.sine_frequency * local_t
            )
            return v_cmd, 0.0
        if local_t < d + z:
            return 0.0, 0.0
        if local_t < 2.0 * d + z:
            w_t = local_t - d - z
            w_cmd = self.args.angular_amplitude * math.sin(
                2.0 * math.pi * self.args.sine_frequency * w_t
            )
            return 0.0, w_cmd
        return 0.0, 0.0

    def _total_duration(self):
        if self.args.mode == 'step':
            return (
                3.0 * self.args.zero_duration +
                2.0 * self.args.step_duration
            )
        return 3.0 * self.args.zero_duration + 2.0 * self.args.sine_duration

    def _timer_callback(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.start_time is None:
            if self._waiting_for_odom(now):
                self._publish_zero()
                return
            self.start_time = now

        t = now - self.start_time
        if t > self._total_duration():
            self._publish_zero()
            self._save_and_finish()
            return

        v_cmd, w_cmd = self._command_at(t)
        cmd = Twist()
        cmd.linear.x = v_cmd
        cmd.angular.z = w_cmd
        self.cmd_pub.publish(cmd)

        v_odom = np.nan
        w_odom = np.nan
        if self.last_odom is not None:
            v_odom = self.last_odom.twist.twist.linear.x
            w_odom = self.last_odom.twist.twist.angular.z

        self.rows.append(
            {
                't': t,
                'v_cmd': v_cmd,
                'w_cmd': w_cmd,
                'v_odom': v_odom,
                'w_odom': w_odom,
            }
        )

    def _waiting_for_odom(self, now):
        if self.last_odom is not None or self.args.wait_for_odom <= 0.0:
            return False

        elapsed = now - self.created_time
        if elapsed > self.args.wait_for_odom:
            self.get_logger().error(
                f'No odom received on {self.args.odom_topic} after '
                f'{self.args.wait_for_odom:.1f} s; aborting test.'
            )
            self.done = True
        return True

    def _publish_zero(self):
        zero = Twist()
        for _ in range(10):
            self.cmd_pub.publish(zero)

    def _save_and_finish(self):
        if self.done:
            return

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        prefix = f'cmd_vel_delay_test_{self.args.mode}_{timestamp}'
        outputs = _save_outputs(
            self.args.output_dir,
            prefix,
            self.rows,
            self.args,
        )
        for output in outputs:
            self.get_logger().info(f'Wrote {output}')
        self.done = True


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description='Apply cmd_vel test signals and record odom response.'
    )
    parser.add_argument('--cmd-topic', default='/cmd_vel')
    parser.add_argument('--odom-topic', default='/odom')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--mode', choices=['step', 'sine'], default='step')
    parser.add_argument('--rate', type=float, default=50.0)
    parser.add_argument('--zero-duration', type=float, default=2.0)
    parser.add_argument('--step-duration', type=float, default=4.0)
    parser.add_argument('--sine-duration', type=float, default=10.0)
    parser.add_argument('--sine-frequency', type=float, default=0.2)
    parser.add_argument('--linear-amplitude', type=float, default=0.3)
    parser.add_argument('--angular-amplitude', type=float, default=0.5)
    parser.add_argument('--max-lag-sec', type=float, default=2.0)
    parser.add_argument(
        '--wait-for-odom',
        type=float,
        default=5.0,
        help='Seconds to wait for odom before starting the test.',
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Run the cmd_vel delay test."""
    argv = sys.argv if argv is None else argv
    cli_args = remove_ros_args(args=argv)[1:]
    args = _parse_args(cli_args)

    rclpy.init(args=argv)
    node = CmdVelDelayTest(args)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node._publish_zero()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
