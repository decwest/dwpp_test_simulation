#!/usr/bin/env python3
"""Plot DWPP velocity and curvature delay profiles from CSV logs."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np


CURVATURE_EPS = 1e-3


def _parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _read_csv_columns(csv_path):
    with csv_path.open('r', newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        if reader.fieldnames is None:
            raise ValueError(f'No CSV header found in {csv_path}')

        columns = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name in columns:
                columns[name].append(row.get(name, ''))
    return columns


def _numeric_series(columns, names):
    for name in names:
        if name in columns:
            return np.array(
                [_parse_float(value) for value in columns[name]],
                dtype=float,
            )
    raise KeyError(f'Missing required column. Tried: {", ".join(names)}')


def _optional_numeric_series(columns, names):
    try:
        return _numeric_series(columns, names)
    except KeyError:
        return None


def _calc_curvature(linear_vel, angular_vel):
    curvature = np.full_like(linear_vel, np.nan, dtype=float)
    valid = np.abs(linear_vel) >= CURVATURE_EPS
    curvature[valid] = angular_vel[valid] / linear_vel[valid]
    return curvature


def _load_time(columns):
    if 't' in columns:
        t = _numeric_series(columns, ['t'])
    elif 'sec' in columns and 'nsec' in columns:
        sec = _numeric_series(columns, ['sec'])
        nsec = _numeric_series(columns, ['nsec'])
        sec_is_integer = np.nanmax(np.abs(sec - np.round(sec))) < 1e-6
        if sec_is_integer:
            t = sec + nsec * 1e-9
        else:
            t = sec
    else:
        first_column = next(iter(columns.values()), [])
        t = np.arange(len(first_column), dtype=float)

    finite = np.isfinite(t)
    if finite.any():
        t = t - t[finite][0]
    return t


def _load_signals(csv_path):
    columns = _read_csv_columns(csv_path)
    t = _load_time(columns)

    v_real = _numeric_series(columns, ['v_real', 'v'])
    w_real = _numeric_series(columns, ['w_real', 'w'])
    v_cmd = _numeric_series(columns, ['v_cmd'])
    w_cmd = _numeric_series(columns, ['w_cmd'])

    kappa_cmd = _optional_numeric_series(columns, ['kappa_cmd'])
    if kappa_cmd is None:
        kappa_cmd = _calc_curvature(v_cmd, w_cmd)

    kappa_odom = _optional_numeric_series(columns, ['kappa_odom'])
    if kappa_odom is None:
        kappa_odom = _calc_curvature(v_real, w_real)

    kappa_reg = _optional_numeric_series(
        columns,
        ['kappa_reg', 'curvature'],
    )
    kappa_pp = _optional_numeric_series(columns, ['kappa_pp'])
    if kappa_pp is None:
        kappa_pp = kappa_reg

    return {
        't': t,
        'v_real': v_real,
        'w_real': w_real,
        'v_cmd': v_cmd,
        'w_cmd': w_cmd,
        'kappa_pp': kappa_pp,
        'kappa_reg': kappa_reg,
        'kappa_cmd': kappa_cmd,
        'kappa_odom': kappa_odom,
    }


def _estimate_delay(t, reference, response, max_lag_sec=3.0, min_points=8):
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


def _plot_curvature(t, signals, output_path):
    plt.figure(figsize=(10, 5))
    curves = [
        ('kappa_pp', 'PP curvature', 'tab:green'),
        ('kappa_reg', 'regulated curvature', 'tab:orange'),
        ('kappa_cmd', 'commanded curvature', 'tab:red'),
        ('kappa_odom', 'odom curvature', 'tab:blue'),
    ]
    for key, label, color in curves:
        values = signals.get(key)
        if values is not None:
            plt.plot(t, values, label=label, color=color, linewidth=1.6)
    plt.xlabel('Time [s]')
    plt.ylabel('Curvature [1/m]')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _write_delay_summary(signals, output_path, max_lag_sec):
    t = signals['t']
    pairs = [
        ('v_cmd -> v_real', signals['v_cmd'], signals['v_real']),
        ('w_cmd -> w_real', signals['w_cmd'], signals['w_real']),
        (
            'kappa_cmd -> kappa_odom',
            signals['kappa_cmd'],
            signals['kappa_odom'],
        ),
    ]

    lines = [
        'Delay summary',
        'Positive delay means the second signal lags the first signal.',
        f'Max lag search window: {max_lag_sec:.3f} s',
        '',
    ]
    for label, reference, response in pairs:
        delay, corr = _estimate_delay(
            t,
            reference,
            response,
            max_lag_sec=max_lag_sec,
        )
        if np.isfinite(delay):
            lines.append(
                f'{label}: delay={delay:.6f} s, '
                f'correlation={corr:.6f}'
            )
        else:
            lines.append(f'{label}: delay=nan s, correlation=nan')

    output_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Plot velocity, angular velocity, and curvature delay profiles '
            'from DWPP CSV logs.'
        )
    )
    parser.add_argument(
        'csv',
        type=Path,
        help='Path to dynamic_window_pure_pursuit_log_*.csv',
    )
    parser.add_argument(
        '-o',
        '--output-dir',
        type=Path,
        default=None,
        help='Directory for outputs. Defaults to the CSV directory.',
    )
    parser.add_argument(
        '--max-lag-sec',
        type=float,
        default=3.0,
        help='Maximum delay window, in seconds, for cross-correlation.',
    )
    return parser.parse_args()


def main():
    """Run the CSV delay analysis command."""
    args = _parse_args()
    if not args.csv.exists():
        raise FileNotFoundError(args.csv)

    output_dir = args.output_dir or args.csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    signals = _load_signals(args.csv)
    t = signals['t']

    _plot_pair(
        t,
        signals['v_cmd'],
        signals['v_real'],
        'Linear Velocity [m/s]',
        output_dir / 'velocity_profile.png',
    )
    _plot_pair(
        t,
        signals['w_cmd'],
        signals['w_real'],
        'Angular Velocity [rad/s]',
        output_dir / 'angular_velocity_profile.png',
    )
    _plot_curvature(t, signals, output_dir / 'curvature_profile.png')
    _write_delay_summary(
        signals,
        output_dir / 'delay_summary.txt',
        args.max_lag_sec,
    )

    print(f'Wrote analysis outputs to {output_dir}')


if __name__ == '__main__':
    main()
