"""Offline step-response measurements; wheel odometry is not external ground truth."""
import csv
import json
import math
from pathlib import Path

import numpy as np

AXES = {'x': 'vx', 'y': 'vy', 'yaw': 'omega'}


def slopes(t, y, window=.1):
    """Local linear fits on actual samples, with no interpolation across gaps."""
    out = np.full(len(t), np.nan)
    for i, centre in enumerate(t):
        mask = np.abs(t-centre) <= window/2 + 1e-9
        tt, yy = t[mask], y[mask]
        if len(tt) < 3 or tt[-1]-tt[0] < window/2 or np.max(np.diff(tt)) > window/2:
            continue
        tt = tt-tt.mean()
        out[i] = np.dot(tt, yy-yy.mean())/np.dot(tt, tt)
    return out


def crossing(t, y, threshold, after, before, consecutive=2):
    """Bracket a sustained crossing by the adjacent original samples."""
    for i in range(1, len(t)-consecutive+1):
        if (after <= t[i] <= before and t[i+consecutive-1] <= before
                and y[i-1] < threshold and np.all(y[i:i+consecutive] >= threshold)):
            f = (threshold-y[i-1])/(y[i]-y[i-1])
            return (float(t[i-1]), float(t[i]), float(t[i-1]+f*(t[i]-t[i-1])))
    return None


def step_metrics(t, velocity, amplitude, hold, *, axis, uncertainty=0., publish_span=0., acceleration_window=.1):
    """Times are relative to publish BEGIN; uncertainty bounds inter-host clocks.

    Onset is threshold-defined, not a fitted pure actuator dead time. Estimated
    t10/t90 are interpolation summaries; their brackets retain sample resolution.
    """
    t, velocity = np.asarray(t, float), np.asarray(velocity, float)
    if (t.shape != velocity.shape or not math.isfinite(amplitude) or amplitude == 0
            or not math.isfinite(hold) or hold <= 0 or uncertainty < 0 or publish_span < 0):
        raise ValueError('Invalid step inputs')
    flags = []
    result = dict(samples=len(t), quality_flags=flags, baseline_velocity=None,
                  steady_velocity=None, steady_gain=None, steady_reached=False,
                  onset_lower_s=None, onset_upper_s=None, onset_threshold=None,
                  t10_s=None, t90_s=None, rise_10_90_s=None,
                  t10_lower_s=None, t10_upper_s=None, t90_lower_s=None, t90_upper_s=None,
                  peak_abs_velocity=None, acceleration_peak=None, acceleration_p95=None,
                  deceleration_peak=None, odom_period_median_s=None, odom_period_p95_s=None,
                  timing_resolution_s=None, onset_resolved=False)
    if len(t) < 10 or not np.isfinite(t).all() or not np.isfinite(velocity).all():
        flags.append('insufficient_or_nonfinite_samples')
        return result
    dt = np.diff(t)
    if np.any(dt <= 0):
        flags.append('nonincreasing_timestamps')
        return result
    result.update(odom_period_median_s=float(np.median(dt)), odom_period_p95_s=float(np.quantile(dt, .95)))
    result['timing_resolution_s'] = result['odom_period_p95_s'] + 2*uncertainty + publish_span
    if dt.max() > max(.1, 3*np.median(dt)):
        flags.append('sampling_gap')
    base = velocity[(t < 0) & (t >= -.8)]
    during = (t >= 0) & (t < hold)
    tail = (t >= .8*hold) & (t < hold)
    if len(base) < 5 or during.sum() < 5 or tail.sum() < 3:
        flags.append('insufficient_baseline_or_plateau')
        return result
    sign = math.copysign(1, amplitude)
    baseline = float(np.median(base))
    noise = float(1.4826*np.median(np.abs(base-baseline)))
    floor = .005 if axis == 'yaw' else .002
    response = sign*(velocity-baseline)
    plateau = float(np.median(response[tail]))
    result.update(baseline_velocity=baseline, steady_velocity=baseline+sign*plateau,
                  steady_gain=plateau/abs(amplitude), peak_abs_velocity=float(np.max(np.abs(velocity[during]))))
    if abs(baseline) > (.01 if axis == 'yaw' else .005):
        flags.append('moving_baseline')
    if plateau <= max(4*noise, 2*floor):
        flags.append('response_below_noise_floor')
        return result
    tt = t[tail]-np.mean(t[tail])
    drift = float(np.dot(tt, response[tail]-np.mean(response[tail]))/np.dot(tt, tt))
    result['steady_reached'] = bool(abs(drift)*(.2*hold) <= max(.05*plateau, 3*noise, floor)
                                    and np.std(response[tail]) <= max(.05*plateau, 3*noise, floor))
    if not result['steady_reached']:
        flags.append('plateau_not_established_in_hold')
    threshold = max(3*noise, floor, .05*plateau)
    result['onset_threshold'] = threshold
    pre = response[(t < -uncertainty) & (t >= -.2)]
    if len(pre) >= 2 and np.any((pre[:-1] >= threshold) & (pre[1:] >= threshold)):
        flags.append('response_before_command_check_baseline_or_clock')
    onset = crossing(t, response, threshold, 0., hold)
    if onset:
        result['onset_lower_s'] = onset[0]-uncertainty-publish_span
        result['onset_upper_s'] = onset[1]+uncertainty
        result['onset_resolved'] = result['onset_lower_s'] > 0 and not flags
    else:
        flags.append('no_sustained_onset')
    for fraction, name in ((.1, 't10'), (.9, 't90')):
        found = crossing(t, response, fraction*plateau, 0., hold)
        if found:
            result[name+'_s'] = found[2]
            result[name+'_lower_s'] = found[0]-uncertainty-publish_span
            result[name+'_upper_s'] = found[1]+uncertainty
    if result['t10_s'] is not None and result['t90_s'] is not None:
        result['rise_10_90_s'] = result['t90_s']-result['t10_s']
    acceleration = slopes(t, response, acceleration_window)
    rising = acceleration[during & np.isfinite(acceleration) & (acceleration > 0)]
    braking = -acceleration[(t >= hold) & np.isfinite(acceleration) & (acceleration < 0)]
    if len(rising):
        result.update(acceleration_peak=float(rising.max()), acceleration_p95=float(np.quantile(rising, .95)))
    if len(braking):
        result['deceleration_peak'] = float(braking.max())
    return result


def read_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def summarize(folder, *, clock_uncertainty_ms=None, acceleration_window=.1):
    """Keep receive-clock and synchronized source-clock results separate."""
    folder = Path(folder)
    session = json.loads((folder/'session.json').read_text())
    pulses = read_csv(folder/'pulses.csv')
    odom = read_csv(folder/'odom.csv')
    if clock_uncertainty_ms is None:
        clock_uncertainty_ms = session['settings'].get('clock_uncertainty_ms')
    if (clock_uncertainty_ms is not None and
            (not math.isfinite(clock_uncertainty_ms) or clock_uncertainty_ms < 0)):
        raise ValueError('Clock uncertainty must be a finite nonnegative upper bound in ms')
    if not math.isfinite(acceleration_window) or acceleration_window <= 0:
        raise ValueError('Acceleration window must be positive')
    modes = ['receive'] + (['source'] if clock_uncertainty_ms is not None else [])
    rows, traces = [], {}
    receive = np.array([float(r['receive_mono_s']) for r in odom])
    source = np.array([float(r['source_ros_s']) for r in odom])
    receive_ros = np.array([float(r['receive_ros_s']) for r in odom])
    source_dt = np.diff(source)
    positive_source_dt = source_dt[source_dt > 0]
    sampling = dict(source_period_median_s=float(np.median(positive_source_dt)) if len(positive_source_dt) else None,
                    source_period_p95_s=float(np.quantile(positive_source_dt, .95)) if len(positive_source_dt) else None,
                    nonincreasing_source_intervals=int(np.count_nonzero(source_dt <= 0)),
                    pc_clock_step_max_s=float(np.max(np.abs(np.diff(receive_ros)-np.diff(receive)))) if len(receive)>1 else None,
                    receive_minus_source_min_s=float(np.min(receive_ros-source)) if len(source) else None,
                    receive_minus_source_max_s=float(np.max(receive_ros-source)) if len(source) else None)
    for pulse in pulses:
        begin, end = float(pulse['window_start_mono_s']), float(pulse['window_end_mono_s'])
        if pulse['status'] != 'completed':
            rows.append(dict(trial_id=pulse['trial_id'], axis=pulse['axis'], status=pulse['status'],
                             time_basis='none', quality_flags=['incomplete_pulse'], error=pulse.get('error', '')))
            continue
        idx = np.flatnonzero((receive >= begin) & (receive <= end))
        speed = np.array([float(odom[i][AXES[pulse['axis']]]) for i in idx])
        for mode in modes:
            if mode == 'receive':
                t = receive[idx]-float(pulse['on_mono_s'])
                hold = float(pulse['off_mono_s'])-float(pulse['on_mono_s'])
                uncertainty = 0.
            else:
                t = source[idx]-float(pulse['on_ros_s'])
                hold = float(pulse['off_ros_s'])-float(pulse['on_ros_s'])
                uncertainty = clock_uncertainty_ms/1000.
            span = float(pulse['on_publish_end_mono_s'])-float(pulse['on_mono_s'])
            m = step_metrics(t, speed, float(pulse['amplitude']), hold, axis=pulse['axis'],
                             uncertainty=uncertainty, publish_span=span, acceleration_window=acceleration_window)
            m.update(trial_id=pulse['trial_id'], axis=pulse['axis'], amplitude=float(pulse['amplitude']),
                     repeat=int(pulse['repeat']), status=pulse['status'], time_basis=mode)
            if mode == 'source' and m['onset_upper_s'] is not None and m['onset_upper_s'] < 0:
                m['quality_flags'].append('response_precedes_command_check_clock_bound')
                m['onset_resolved'] = False
            rows.append(m)
            if mode == 'receive':
                traces[pulse['trial_id']] = (pulse, t, speed, hold)
    groups = []
    for axis in AXES:
        chosen = [r for r in rows if r.get('axis') == axis and r.get('time_basis') == 'receive']
        if not chosen:
            continue
        g = dict(axis=axis, completed_pulses=len(chosen),
                 note='Observed under the tested command levels and loads; not a certified physical maximum')
        for key in ('peak_abs_velocity', 'acceleration_peak', 'deceleration_peak'):
            values = [r[key] for r in chosen if r.get(key) is not None]
            g['observed_'+key] = max(values) if values else None
        groups.append(g)
    repeated = []
    for axis, amplitude, basis in sorted({(r['axis'], r['amplitude'], r['time_basis']) for r in rows if 'amplitude' in r}):
        chosen = [r for r in rows if (r.get('axis'), r.get('amplitude'), r['time_basis']) == (axis, amplitude, basis)]
        item = dict(axis=axis, amplitude=amplitude, time_basis=basis, recorded=len(chosen),
                    onset_resolved_trials=sum(r['onset_resolved'] for r in chosen),
                    quality_flagged_trials=sum(bool(r['quality_flags']) for r in chosen))
        for key in ('onset_lower_s', 'onset_upper_s', 'rise_10_90_s', 'steady_velocity',
                    'steady_gain', 'peak_abs_velocity', 'acceleration_peak', 'acceleration_p95', 'deceleration_peak'):
            values = [r[key] for r in chosen if r[key] is not None]
            item[key+'_n'] = len(values)
            item[key+'_mean'] = float(np.mean(values)) if values else None
            item[key+'_sd'] = float(np.std(values, ddof=1)) if len(values)>1 else None
        repeated.append(item)
    report = dict(session_status=session['status'], pulse_rows=len(pulses), metrics=rows, observed_maxima=groups,
                  repeat_groups=repeated,
                  clock_uncertainty_ms=clock_uncertainty_ms, acceleration_window_s=acceleration_window, sampling=sampling,
                  note='Receive timing includes outbound/inbound transport and executor latency. Source timing requires a justified inter-host clock bound; it still includes outbound transport, onboard control and encoder estimation. Threshold onset is not pure motor dead time. Preserve sample brackets; no sub-sample precision claim. Direct cmd_vel bypasses Nav2 velocity_smoother, but retains onboard limits/filters.')
    destination = folder/'analysis'
    destination.mkdir(exist_ok=True)
    (destination/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    if rows:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with (destination/'metrics.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
    if repeated:
        with (destination/'group_metrics.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(repeated[0]))
            writer.writeheader(); writer.writerows(repeated)
    lines = ['# Step response observations', '', report['note'], '',
             '| Axis | Observed peak speed | Observed peak acceleration | Observed peak deceleration |',
             '|---|---:|---:|---:|']
    for g in groups:
        values = [g.get('observed_'+key) for key in ('peak_abs_velocity', 'acceleration_peak', 'deceleration_peak')]
        lines.append('| '+g['axis']+' | '+' | '.join('unavailable' if v is None else f'{v:.5g}' for v in values)+' |')
    lines += ['', 'Translation uses m/s and m/s²; yaw uses rad/s and rad/s².',
              'See metrics.csv for per-pulse onset bounds, t10/t90, plateau gain and quality flags.',
              'A steady plateau below the command is evidence to inspect, not proof of a hardware speed limit.',
              'Changing the command levels or acceleration-fit window changes the scope of these observations.']
    (destination/'report.md').write_text('\n'.join(lines)+'\n')
    # Numeric results and the report remain usable if plotting fails.
    plot_traces(traces, destination, acceleration_window)
    return report


def plotting_backend():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def check_plotting():
    """Exercise the actual PNG/PDF dependencies before any robot handoff/motion."""
    from io import BytesIO
    plt = plotting_backend()
    fig, ax = plt.subplots(figsize=(1, 1))
    try:
        ax.plot([0, 1], [0, 1])
        for suffix in ('png', 'pdf'):
            with BytesIO() as output:
                fig.savefig(output, format=suffix)
    finally:
        plt.close(fig)


def plot_traces(traces, destination, acceleration_window):
    plt = plotting_backend()
    for axis in AXES:
        selected = [v for v in traces.values() if v[0]['axis'] == axis]
        if not selected:
            continue
        fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True, constrained_layout=True)
        for pulse, t, v, hold in selected:
            label = f"{float(pulse['amplitude']):+.3g}, r{pulse['repeat']}"
            line, = ax[0].plot(t, v, label=label, alpha=.7, linewidth=1)
            ax[0].plot([min(t), 0, 0, hold, hold, max(t)],
                       [0, 0, float(pulse['amplitude']), float(pulse['amplitude']), 0, 0],
                       '--', color=line.get_color(), alpha=.3)
            ax[1].plot(t, slopes(t, v, acceleration_window), color=line.get_color(), alpha=.7)
        for a in ax:
            a.axvline(0, color='k', linewidth=.8); a.grid(alpha=.2)
        units = 'rad' if axis == 'yaw' else 'm'
        ax[0].set(ylabel=f'Velocity [{units}/s]', title=f'{axis}: wheel odometry; PC receive timing')
        ax[1].set(xlabel='Time from step publication [s]', ylabel=f'Acceleration [{units}/s²]')
        ax[0].legend(fontsize=6, ncol=3)
        for suffix in ('png', 'pdf'):
            fig.savefig(destination/f'response_{axis}.{suffix}', dpi=160)
        plt.close(fig)
