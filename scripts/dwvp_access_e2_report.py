#!/usr/bin/env python3
"""Export route-design figures and per-repeat statistics from isolated E2 outputs.

Reads recorded data only. Run on the host; it never starts ROS or Docker.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import tarfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D

from dwvp_access_metrics import project_tracking
from dwvp_access_path import load_map, clearance_profile

METHODS = ['VP_CLIP', 'DWVP', 'DWVP_fixed', 'MPPI', 'DWB']
LABELS = dict(VP_CLIP='VP_CLIP', DWVP='DWVP adaptive', DWVP_fixed='DWVP fixed 0.11 m', MPPI='MPPI', DWB='DWB')
COLORS = dict(zip(METHODS, ['#0072B2', '#009E73', '#56B4E9', '#E69F00', '#CC79A7']))
METRICS = ['eval_mean_position_error_m', 'eval_max_position_error_m',
           'eval_mean_heading_error_deg', 'eval_max_heading_error_deg',
           'travel_time_s', 'compute_time_mean_ms']


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    return np.atleast_1d(np.genfromtxt(path, delimiter=',', names=True))


def host_path(value, work):
    path = Path(value)
    return work/path.relative_to('/work') if path.is_relative_to('/work') else path


def statistics(rows):
    out = []
    for method in METHODS:
        samples = [r for r in rows if r.get('variant', r['controller']) == method]
        group = dict(method=method, repeats=len(samples), successes=sum(r['success'] for r in samples),
                     evaluation_complete=sum(r.get('evaluation_complete', False) for r in samples),
                     data_error_runs=sum(bool(r.get('data_errors')) for r in samples),
                     failed_compute_calls=sum(r.get('timing', {}).get('failed_calls', 0) for r in samples))
        for key in METRICS:
            values = [r[key] for r in samples if r.get(key) is not None and math.isfinite(r[key])]
            group[key] = dict(n=len(values), mean=float(np.mean(values)) if values else None,
                              sd=float(np.std(values, ddof=1)) if len(values) > 1 else None)
        out.append(group)
    return out


def format_stat(stat, digits):
    if not stat['n']:
        return '—'
    mean = f"{stat['mean']:.{digits}f}"
    sd = f"{stat['sd']:.{digits}f}" if stat['sd'] is not None else '—'
    return f'{mean} ± {sd}' + (f" (n={stat['n']})" if stat['n'] != 3 else '')


def table(groups):
    lines = ['| 手法 | 成功 | 位置平均 [m] | 位置最大 [m] | 方位平均 [deg] | 方位最大 [deg] | 走行時間 [s] | 計算時間 [ms] |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for group in groups:
        values = [format_stat(group[key], 4 if 'position' in key else 2 if 'heading' in key or key == 'travel_time_s' else 3) for key in METRICS]
        lines.append('| '+ ' | '.join([LABELS[group['method']], f"{group['successes']}/{group['repeats']}", *values])+' |')
    return '\n'.join(lines)


def export_route(name, route, check, sweep, verify, work, output):
    dest = output/name
    dest.mkdir(parents=True, exist_ok=False)
    shutil.copytree(route, dest/'route')
    shutil.copytree(check, dest/'check')
    path = np.loadtxt(route/'E2_environment.csv', delimiter=',', skiprows=1)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1))]
    meta = read_json(route/'path_metadata.json')
    checks = read_json(check/'report.json')
    sweep_report = read_json(sweep/'report.json')
    verify_report = read_json(verify/'report.json')
    rows = verify_report['rows']
    groups = statistics(rows)
    completion = {g['method']: f"{g['successes']}/{g['repeats']} completed" for g in groups}
    (dest/'statistics.json').write_text(json.dumps(groups, indent=2, allow_nan=False)+'\n')
    (dest/'results_table.md').write_text(table(groups)+'\n')
    for source, subdir in [(sweep, 'sweep'), (verify, 'verify')]:
        with tarfile.open(dest/(subdir+'_raw.tar.gz'), 'w:gz') as archive:
            archive.add(source, arcname=source.name)
        for filename in ('report.json', 'plan_before_results.json'):
            shutil.copy2(source/filename, dest/(subdir+'_'+filename))
    info, pixels, blocked = load_map(route/'map.yaml')
    occupancy = pixels/255 if info.get('negate', 0) else (255-pixels)/255
    occupied = np.flipud(occupancy > info['occupied_thresh'])
    fig, ax = plt.subplots(figsize=(9, 9))
    h, w = pixels.shape; res = info['resolution']; ox, oy, yaw = info['origin']
    transform = Affine2D().rotate(yaw).translate(ox, oy)+ax.transData
    ax.imshow(pixels, cmap='gray', vmin=0, vmax=255, origin='upper',
              extent=[0, w*res, 0, h*res], transform=transform)
    ax.plot(path[:, 0], path[:, 1], 'k--', lw=2.5, label='Reference path')
    for interval in meta['facing']['intervals']:
        selected = (arc >= interval['start_m']) & (arc <= interval['end_m'])
        ax.plot(path[selected, 0], path[selected, 1], color='#D55E00', lw=4, label='Facing shelf / table')
        for lo, hi in [(interval['transition_start_m'], interval['start_m']),
                       (interval['end_m'], interval['transition_end_m'])]:
            mask = (arc >= lo) & (arc <= hi)
            ax.plot(path[mask, 0], path[mask, 1], color='#D55E00', lw=3, alpha=.4)
    step = max(1, len(path)//25)
    ax.quiver(path[::step, 0], path[::step, 1], np.cos(path[::step, 2]), np.sin(path[::step, 2]),
              scale_units='xy', angles='xy', scale=6, width=.003, color='black')
    profiles, axes = plt.subplots(2, len(METHODS), figsize=(3.8*len(METHODS), 7), sharex=True, sharey='row')
    observed_clearances = []
    for row in rows:
        method = row.get('variant', row['controller'])
        if method not in METHODS:
            continue
        folder = host_path(row['session'], work)/'runs'/row['trial_id']
        tracking = read_csv(folder/'tracking.csv')
        valid = np.isfinite(np.c_[tracking['x'], tracking['y'], tracking['yaw'], tracking['stamp_s']]).all(axis=1)
        tracking = tracking[valid]
        poses = np.c_[tracking['x'], tracking['y'], tracking['yaw']]
        projection = project_tracking(poses, path)
        distances, cells = clearance_profile(poses, info, occupied)
        smallest = int(np.argmin(distances))
        observed_clearances.append(dict(method=method, repeat=row['repeat'],
            centre_to_occupied_m=float(distances[smallest]), footprint_margin_m=float(distances[smallest]-.22),
            position=poses[smallest, :2].tolist(), nearest_cell=cells[smallest].tolist(),
            note=('Post hoc clearance of synthetic tracking with supplied static map and ideal ray scans; not physical sensing'
                  if verify_report['plan'].get('map_yaml_sha256') else
                  'Post hoc map overlay of free-map synthetic tracking; not sensed collision avoidance')))
        color = COLORS[method]
        style = '--' if method == 'DWVP_fixed' or not row['success'] else '-'
        label = f"{LABELS[method]} ({completion[method]})" if row['repeat'] == 1 else None
        ax.plot(poses[:, 0], poses[:, 1], style, color=color, alpha=.6, lw=1, label=label)
        if not row['success']:
            ax.scatter(*poses[-1, :2], marker='x', color=color, s=35, zorder=9)
        column = METHODS.index(method)
        axes[0, column].plot(projection[:, 2], np.abs(np.rad2deg(projection[:, 1])), color=color, lw=1, alpha=.6)
        odom = read_csv(folder/'odom.csv')
        mask = (odom['stamp_s'] >= tracking['stamp_s'][0]) & (odom['stamp_s'] <= tracking['stamp_s'][-1])
        odom = odom[mask]
        progress = np.interp(odom['stamp_s'], tracking['stamp_s'], projection[:, 2])
        axes[1, column].plot(progress, np.hypot(odom['vx'], odom['vy']), color=color, lw=1, alpha=.6)
    ax.scatter(*path[0, :2], c='green', s=55, zorder=10, label='Start')
    ax.scatter(*path[-1, :2], c='red', marker='x', s=70, zorder=10, label='Goal')
    ax.set(xlabel='Map x [m]', ylabel='Map y [m]', title=name+' | synthetic tracking, 3 repeats per variant',
           xlim=(path[:, 0].min()-.9, path[:, 0].max()+.9), ylim=(path[:, 1].min()-.9, path[:, 1].max()+.9))
    ax.set_aspect('equal'); ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(.5, -.08), ncol=3)
    fig.tight_layout(); fig.savefig(dest/'map_tracking.png', dpi=180); plt.close(fig)
    for column, method in enumerate(METHODS):
        axes[0, column].set_title(f"{LABELS[method]} ({completion[method]})")
        axes[1, column].axhline(.22, color='black', ls=':', lw=.8)
        axes[1, column].set_xlabel('Projected arc length [m]')
        for row_index in range(2):
            a = axes[row_index, column]
            a.set_xlim(0, arc[-1]); a.grid(alpha=.25)
            a.axvline(arc[-1]-.6, color='gray', ls='--', lw=.8)
            for interval in meta['facing']['intervals']:
                a.axvspan(interval['start_m'], interval['end_m'], color='#D55E00', alpha=.1)
                a.axvline(interval['transition_start_m'], color='#D55E00', ls=':', lw=.8)
                a.axvline(interval['transition_end_m'], color='#D55E00', ls=':', lw=.8)
    axes[0, 0].set_ylabel('Absolute heading error [deg]')
    axes[1, 0].set_ylabel('Odometry translation speed [m/s]')
    profiles.suptitle(name+' | all 3 repeats; shaded: fixed facing; dotted orange: transitions; gray: evaluation end')
    profiles.tight_layout(); profiles.savefig(dest/'heading_speed.png', dpi=160); plt.close(profiles)
    (dest/'tracking_map_clearance.json').write_text(json.dumps(observed_clearances, indent=2)+'\n')
    nearest_via = []
    for via in meta['via']:
        index = int(np.argmin(np.linalg.norm(path[:, :2]-via, axis=1)))
        nearest_via.append(dict(requested=via, nearest_smoothed_xy=path[index, :2].tolist(),
                                displacement_m=float(np.linalg.norm(path[index, :2]-via))))
    facts = dict(route=name, checks=checks, metadata=meta, via=nearest_via,
                 selected=sweep_report['selected'], groups=groups, observed_clearances=observed_clearances,
                 raw_locations=dict(route=str(route), check=str(check), sweep=str(sweep), verify=str(verify)))
    (dest/'facts.json').write_text(json.dumps(facts, indent=2, allow_nan=False)+'\n')
    return facts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--route', nargs=5, action='append', required=True,
                        metavar=('NAME', 'ROUTE_DIR', 'CHECK_DIR', 'SWEEP_DIR', 'VERIFY_DIR'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    facts = [export_route(name, *(Path(p) for p in directories), args.work, args.output)
             for name, *directories in args.route]
    (args.output/'all_facts.json').write_text(json.dumps(facts, indent=2, allow_nan=False)+'\n')
    hashes = {str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in args.output.rglob('*') if p.is_file()}
    (args.output/'SHA256SUMS.json').write_text(json.dumps(hashes, indent=2)+'\n')


if __name__ == '__main__':
    main()
