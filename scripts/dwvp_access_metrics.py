"""Offline metrics. Missing, stale and incomplete attempts remain in the report."""
import csv
import json
import math
from pathlib import Path

import numpy as np
import yaml

AXES = ('vx', 'vy', 'omega')
COMMON_METRICS = ('eval_max_position_error_m', 'eval_mean_position_error_m',
                  'eval_position_error_integral_m_s', 'eval_max_heading_error_deg',
                  'eval_mean_heading_error_deg', 'eval_heading_error_integral_deg_s',
                  'constraint_violation_pct', 'travel_time_s', 'compute_time_mean_ms')


def transient_metrics(task):
    if task == 'E1_lateral':
        return ('crossing_m',)
    if task in ('E1_orientation_nominal', 'E1_orientation_half'):
        return ('transition_heading_lag_deg', 'post_transition_heading_overshoot_deg')
    return ()


def project_tracking(poses, path):
    """Distance, signed yaw, and projected arc length using the same segment.

    Extend progress outside the endpoints so a pose behind the start is excluded
    from a start-to-goal evaluation window. Distances still use finite segments.
    """
    from dwvp_access_experiment import wrap
    delta = np.diff(path[:, :2], axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    arc = np.r_[0., np.cumsum(lengths)]
    output = np.full((len(poses), 3), np.nan)
    for i, pose in enumerate(poses):
        if not np.isfinite(pose).all():
            continue
        fractions = np.sum((pose[:2] - path[:-1, :2]) * delta, axis=1) / lengths**2
        clipped = np.clip(fractions, 0., 1.)
        distances = np.linalg.norm(path[:-1, :2] + clipped[:, None]*delta - pose[:2], axis=1)
        k = int(np.argmin(distances))
        yaw = path[k, 2] + clipped[k]*wrap(path[k+1, 2] - path[k, 2])
        progress = arc[k] + clipped[k]*lengths[k]
        if (k == 0 and fractions[k] < 0) or (k == len(delta)-1 and fractions[k] > 1):
            progress = arc[k] + fractions[k]*lengths[k]
        output[i] = distances[k], wrap(pose[2] - yaw), progress
    return output


def tracking_metrics(times, poses, path, condition, initial_pose, valid=None):
    """Simulator-aligned spatial window and adjacent-sample trapezoids.

    No boundary interpolation or bridging an excluded/stale sample. A singleton
    has zero integral and no time mean; an empty window has no error metrics.
    """
    from dwvp_access_experiment import wrap
    times, poses, path = np.asarray(times), np.asarray(poses), np.asarray(path)
    projected = project_tracking(poses, path)
    position, signed_yaw, progress = projected.T
    heading = np.abs(np.rad2deg(signed_yaw))
    window = condition['evaluation']
    end = float(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1).sum() - window['goal_margin_m'])
    if end <= window['start_m']:
        raise ValueError('Evaluation window is empty: path must exceed goal margin and start')
    finite = np.isfinite(projected).all(axis=1) & np.isfinite(times)
    if valid is not None:
        finite &= valid
    mask = finite & (progress >= window['start_m'] - 1e-10) & (progress <= end)
    dt = np.diff(times)
    adjacent = mask[:-1] & mask[1:] & np.isfinite(dt) & (dt > 0)
    duration = float(dt[adjacent].sum())
    out = dict(evaluation_start_m=window['start_m'], evaluation_end_m=end,
               evaluation_complete=bool(np.any(finite & (progress >= end))),
               evaluation_samples=int(mask.sum()), eval_duration_s=duration,
               invalid_time_intervals=int(np.count_nonzero(~np.isfinite(dt) | (dt <= 0))),
               crossing_m=None, transition_heading_lag_deg=None,
               post_transition_heading_overshoot_deg=None)
    for name, unit, errors in (('position', 'm', position), ('heading', 'deg', heading)):
        integral = float((.5*(np.abs(errors[:-1]) + np.abs(errors[1:])))[adjacent] @ dt[adjacent])
        out['eval_max_'+name+'_error_'+unit] = float(errors[mask].max()) if mask.any() else None
        out['eval_mean_'+name+'_error_'+unit] = integral/duration if duration > 0 else None
        out['eval_'+name+'_error_integral_'+unit+'_s'] = integral if mask.any() else None
    if 'length_m' in condition and 'orientation_length_m' not in condition and condition.get('start_pose') is not None and abs(condition['start_pose'][1]) > 0:
        direction = path[-1, :2] - path[0, :2]
        direction /= np.linalg.norm(direction)
        normal = np.array([-direction[1], direction[0]])
        initial = (np.asarray(initial_pose)[:2] - path[0, :2]) @ normal
        lateral = (poses[:, :2] - path[0, :2]) @ normal
        if mask.any():
            out['crossing_m'] = max(0., float((-np.sign(initial)*lateral[mask]).max()))
    if 'orientation_length_m' in condition:
        begin = condition['orientation_start_m']
        ramp_end = begin + condition['orientation_length_m']
        sign = np.sign(condition['goal_yaw_rad'])
        changing = mask & (progress >= begin - 1e-10) & (progress <= ramp_end + 1e-10)
        after = mask & (progress > ramp_end + 1e-10)
        if changing.any():
            out['transition_heading_lag_deg'] = max(0., float((-sign*np.rad2deg(signed_yaw[changing])).max()))
        if after.any():
            out['post_transition_heading_overshoot_deg'] = max(0., float((sign*np.rad2deg(wrap(poses[after, 2] - path[-1, 2]))).max()))
    return out, projected, mask


def constraint_rates(out, threshold):
    total = out['constraint_total_samples']
    known = out['constraint_evaluable_samples']
    violations = out['constraint_violation_samples']
    unknown = out['constraint_unknown_samples']
    out.update(constraint_violation_pct=100.*violations/known if known else None,
               constraint_unknown_pct=100.*unknown/total if total else None,
               constraint_violation_lower_pct=100.*violations/total if total else None,
               constraint_violation_upper_pct=100.*(violations+unknown)/total if total else None,
               constraint_unknown_flag=bool(total and 100.*unknown/total > threshold),
               constraint_no_evaluable_cycles=known == 0,
               constraint_unknown_threshold_pct=threshold)
    return out


def read_events(path, columns):
    if not path.exists():
        return np.empty(0, dtype=[(name, float) for name in columns]), 'missing'
    try:
        data = np.atleast_1d(np.genfromtxt(path, delimiter=',', names=True, dtype=float))
        if data.dtype.names is None or not set(columns) <= set(data.dtype.names):
            raise ValueError('missing columns')
        return data, None
    except (ValueError, OSError) as exc:
        return np.empty(0, dtype=[(name, float) for name in columns]), str(exc)


def command_metrics(folder, result, common, max_age=.2, unknown_threshold=5.):
    out = {'status': 'missing', 'samples': 0, 'constraint_violation_pct': None,
           'constraint_total_samples': 0, 'constraint_evaluable_samples': 0,
           'constraint_violation_samples': 0, 'constraint_unknown_samples': 0}
    start, duration = result.get('start_stamp_s'), result.get('duration_s')
    if start is None or duration is None:
        return constraint_rates(out, unknown_threshold)
    raw, raw_error = read_events(folder / 'raw.csv', ('stamp_s', *AXES))
    applied, applied_error = read_events(folder / 'applied.csv', ('stamp_s', *AXES))
    out['raw_file_error'], out['applied_file_error'] = raw_error, applied_error
    invalid_stamps = int(np.count_nonzero(~np.isfinite(raw['stamp_s'])))
    raw = raw[(raw['stamp_s'] >= start) & (raw['stamp_s'] <= start+duration)]
    out['constraint_total_samples'] = len(raw) + invalid_stamps
    raw_good = np.isfinite(np.column_stack([raw['stamp_s'], *[raw[k] for k in AXES]])).all(axis=1)
    out['invalid_raw_samples'] = int((~raw_good).sum()) + invalid_stamps
    out['constraint_unknown_samples'] = out['constraint_total_samples']
    raw = raw[raw_good]
    applied_good = np.isfinite(np.column_stack([applied['stamp_s'], *[applied[k] for k in AXES]])).all(axis=1)
    out['invalid_applied_samples'] = int((~applied_good).sum())
    applied_bad_stamp = bool(np.any(~np.isfinite(applied['stamp_s'])))
    # Retain invalid values at known timestamps: they must block substitution
    # of an older, valid command for the actual preceding smoother output.
    applied = applied[np.isfinite(applied['stamp_s'])]
    if not len(raw):
        return constraint_rates(out, unknown_threshold)
    clock = 'receive_monotonic_s' if all('receive_monotonic_s' in x.dtype.names for x in (raw, applied)) else 'stamp_s'
    out['pairing_clock'] = clock
    if (not np.isfinite(raw[clock]).all() or not np.isfinite(applied[clock]).all()
            or np.any(np.diff(raw[clock]) < 0) or np.any(np.diff(applied[clock]) < 0)):
        out['status'] = 'nonmonotonic_timestamps'
        return constraint_rates(out, unknown_threshold)
    frequency = common['control_frequency_hz']
    values = np.c_[[raw[k] for k in AXES]].T
    lo, hi = np.asarray(common['min_velocity']), np.asarray(common['max_velocity'])
    scale = np.maximum(abs(lo), abs(hi))
    vel_excess = np.maximum(np.maximum(values-hi, lo-values), 0)/scale
    # Strictly earlier received smoother output; never substitute previous raw.
    indices = np.searchsorted(applied[clock], raw[clock], side='left')-1
    paired = indices >= 0
    out['prior_applied_absent_samples'] = int((~paired).sum())
    out['prior_applied_stale_samples'] = 0
    if len(applied):
        ages = raw[clock] - applied[clock][np.maximum(indices, 0)]
        out['prior_applied_stale_samples'] = int(np.count_nonzero(paired & (ages > max_age)))
        paired &= ages <= max_age
        preceding_finite = np.isfinite(np.column_stack([applied[k][np.maximum(indices,0)] for k in AXES])).all(axis=1)
        paired &= preceding_finite
    if applied_bad_stamp:
        paired[:] = False
    out.update(status='available', samples=len(raw), prior_applied_missing_or_stale=int((~paired).sum()))
    increments = np.empty((0, 3))
    if paired.any():
        previous = np.c_[[applied[k][indices[paired]] for k in AXES]].T
        delta = values[paired]-previous
        accel, decel = np.asarray(common['max_accel']), np.asarray(common['max_decel'])
        # Directional one-cycle command increments, not physical acceleration.
        increments = np.maximum(np.where(delta >= 0, delta*frequency/accel, delta*frequency/decel)-1, 0)
    # A cycle is counted once if ANY component exceeds EITHER bound.
    violates = np.any(vel_excess > 1e-6, axis=1)
    violates[paired] |= np.any(increments > 1e-6, axis=1)
    # A speed violation is conclusive even when the increment is unavailable.
    evaluable = paired | violates
    unknown = int((~evaluable).sum()) + out['invalid_raw_samples']
    out.update(constraint_violation_samples=int(violates.sum()),
               constraint_evaluable_samples=int(evaluable.sum()), constraint_unknown_samples=unknown)
    constraint_rates(out, unknown_threshold)
    out['constraint_rate_definition'] = (
        '100 * violating cycles / classifiable received cycles; bounds use all received cycles. '
        'Null only when the denominator is zero. Reference: strictly earlier fresh smoother output.')
    for i, axis in enumerate(AXES):
        out[axis+'_velocity_excess_ratio_max'] = float(vel_excess[:, i].max())
        out[axis+'_velocity_excess_time_s'] = float(np.count_nonzero(vel_excess[:, i] > 1e-6)/frequency)
        out[axis+'_increment_excess_ratio_max'] = float(increments[:, i].max()) if len(increments) else None
        out[axis+'_increment_excess_time_s'] = float(np.count_nonzero(increments[:, i] > 1e-6)/frequency) if len(increments) else None
    # Output-side comparison: pair each smoother message with its most recent raw input.
    active = applied[(applied['stamp_s'] >= start) & (applied['stamp_s'] <= start+duration)]
    active_finite = np.isfinite(np.column_stack([active[k] for k in AXES])).all(axis=1)
    indices = np.searchsorted(raw[clock], active[clock], side='right')-1
    paired = indices >= 0
    if len(active):
        paired &= active[clock] - raw[clock][np.maximum(indices, 0)] <= max_age
        paired &= active_finite
    out['smoother_difference_pairs'] = int(paired.sum())
    out['smoother_difference_unpaired'] = int((~paired).sum())
    for axis in AXES:
        difference = active[axis][paired] - raw[axis][indices[paired]]
        out[axis+'_smoother_difference_rmse'] = float(np.sqrt(np.mean(difference**2))) if len(difference) else None
        out[axis+'_smoother_difference_max'] = float(np.max(np.abs(difference))) if len(difference) else None
    out['receive_interval_max_s'] = float(np.max(np.diff(raw['stamp_s']))) if len(raw)>1 else None
    if clock == 'receive_monotonic_s':
        out['monotonic_receive_interval_max_s'] = float(np.max(np.diff(raw[clock]))) if len(raw)>1 else None
        out['receive_clock_step_samples'] = int(np.count_nonzero(np.abs(np.diff(raw['stamp_s'])-np.diff(raw[clock])) > .1))
    out['duration_definition'] = 'Exceeding controller samples / configured frequency; reception gaps are not filled.'
    return out


def timing_metrics(folder, result, controller, raw_samples):
    out = {'samples': 0, 'failed_calls': 0, 'sequence_gaps': 0, 'invalid_samples': 0,
           'mean_ms': None, 'p95_ms': None, 'max_ms': None, 'status': 'missing'}
    start, duration = result.get('start_stamp_s'), result.get('duration_s')
    if start is None or duration is None or not (folder / 'timing.csv').exists():
        return out
    rows = []
    try:
        with (folder / 'timing.csv').open() as stream:
            for item in csv.DictReader(stream):
                try:
                    stamp = float(item['stamp_s'])
                    if item['controller'] != controller or not start <= stamp <= start+duration:
                        continue
                    ms, seq = float(item['compute_time_ms']), int(item['sequence'])
                    if not math.isfinite(ms) or ms < 0 or item['success'] not in ('true', 'false'):
                        raise ValueError('invalid timing sample')
                    rows.append((seq, ms, item['success'] == 'true'))
                except (ValueError, KeyError):
                    out['invalid_samples'] += 1
    except OSError:
        return out
    if rows:
        times = np.array([r[1] for r in rows])
        delta = np.diff([r[0] for r in rows])
        out.update(samples=len(rows), failed_calls=sum(not r[2] for r in rows),
                   sequence_gaps=int(np.maximum(delta-1, 0).sum()),
                   sequence_nonincreasing=int(np.count_nonzero(delta <= 0)),
                   mean_ms=float(times.mean()), p95_ms=float(np.percentile(times, 95)), max_ms=float(times.max()),
                   status='available')
    out['raw_minus_successful_timing_samples'] = raw_samples - sum(r[2] for r in rows)
    return out


def obstacle_distance(xy, obstacles):
    """Distance from robot centre to surveyed circle/polygon surfaces, signed inside."""
    all_distances = []
    for obstacle in obstacles:
        if obstacle['type'] == 'circle':
            all_distances.append(np.linalg.norm(xy-np.asarray(obstacle['center']), axis=1)-obstacle['radius'])
        elif obstacle['type'] == 'polygon':
            vertices = np.asarray(obstacle['vertices'], dtype=float)
            closest = np.full(len(xy), np.inf)
            inside = np.zeros(len(xy), dtype=bool)
            for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
                d = b-a
                if np.dot(d, d) <= 0:
                    raise ValueError('Duplicate polygon vertex')
                t = np.clip((xy-a)@d/np.dot(d,d), 0, 1)
                closest = np.minimum(closest, np.linalg.norm(xy-(a+t[:,None]*d), axis=1))
                if b[1] != a[1]:
                    inside ^= ((a[1]>xy[:,1]) != (b[1]>xy[:,1])) & (xy[:,0] < (b[0]-a[0])*(xy[:,1]-a[1])/(b[1]-a[1])+a[0])
            all_distances.append(np.where(inside, -closest, closest))
        else:
            raise ValueError('Survey geometry supports circle and polygon')
    return np.min(all_distances, axis=0)


def summarize_session(session, tracking_errors):
    session = Path(session)
    manifest = json.loads((session/'manifest.json').read_text())
    config = yaml.safe_load((session/manifest['config_file']).read_text())
    from dwvp_access_experiment import condition_common
    settings = config['metrics']
    max_age = settings['maximum_source_age_s']
    rows, diagnostics, timings = [], {}, {}
    for trial in manifest['trials']:
        common = condition_common(config, trial['task'])
        folder = session/'runs'/trial['id']
        if not folder.exists():
            continue
        errors = []
        result = {'status': 'incomplete', 'success': False}
        try:
            result.update(json.loads((folder/'result.json').read_text()))
        except (OSError, ValueError) as exc:
            errors.append('result.json: '+str(exc))
        data, issue = read_events(folder/'tracking.csv', ('t','stamp_s','x','y','yaw','tf_age_s','raw_age_s','applied_age_s','odom_age_s','odom_source_age_s'))
        if issue:
            errors.append('tracking.csv: '+issue)
        valid = np.isfinite(np.c_[data['x'],data['y'],data['yaw']]).all(axis=1)
        fresh = valid.copy()
        for key in ('tf_age_s','odom_age_s','odom_source_age_s'):
            fresh &= (data[key]>=0) & (data[key]<=max_age)
        commands_fresh = np.ones(len(data), dtype=bool)
        for key in ('raw_age_s','applied_age_s'):
            commands_fresh &= (data[key]>=0) & (data[key]<=max_age)
        quality = fresh & commands_fresh
        grace = fresh & (data['t'] <= 1/manifest['control_frequency_hz'])
        missing_prefix = result.get('missing_command_prefix_s')
        if missing_prefix is None:
            ready = np.flatnonzero(commands_fresh)
            missing_prefix = float(data['t'][ready[0]]) if len(ready) else result.get('duration_s')
        row = dict(trial_id=trial['id'], task=trial['task'], controller=trial['controller'], repeat=trial['repeat'],
                   status=result['status'], success=bool(result.get('success',False)), duration_s=result.get('duration_s'),
                   samples=len(data), valid_pose_samples=int(valid.sum()), fresh_pose_samples=int(fresh.sum()),
                   stale_or_missing_samples=int((~quality).sum()), invalid_after_warmup_samples=int((~(quality|grace)).sum()),
                   missing_command_prefix_s=missing_prefix,
                   yaw_error_role='reference_only' if trial['controller'] in ('RPP','DWPP') else 'tracking',
                   position_rmse_m=None, position_max_m=None, yaw_rmse_rad=None, yaw_max_rad=None,
                   yaw_rmse_deg=None, yaw_max_deg=None, lateral_initial_error_m=None,
                   lateral_crossing_raw_m=None, lateral_crossing_beyond_deadband_m=None,
                   lateral_convergence_time_s=None, lateral_convergence_distance_m=None,
                   yaw_lead_max_rad=None, yaw_lag_max_rad=None,
                   orientation_min_speed_m_s=None, orientation_predicted_speed_m_s=None,
                   orientation_speed_samples=0, orientation_missing_speed_samples=0,
                   acceleration_scale=trial['acceleration_scale'], final_position_error_m=result.get('final_position_error_m'),
                   final_yaw_error_rad=result.get('final_yaw_error_rad'), obstacle_near_mean_speed_m_s=None,
                   obstacle_near_samples=0, obstacle_near_missing_or_stale_samples=len(data) if trial['task']=='E2_environment' else 0, surveyed_min_clearance_m=None)
        row.update({key: None for key in COMMON_METRICS})
        row.update(crossing_m=None, transition_heading_lag_deg=None, post_transition_heading_overshoot_deg=None,
                   evaluation_start_m=None, evaluation_end_m=None, evaluation_complete=False,
                   evaluation_samples=0, eval_duration_s=0., invalid_time_intervals=0)
        row['travel_time_s'] = result.get('duration_s') if row['success'] else None
        try:
            path = np.atleast_2d(np.loadtxt(folder/'reference.csv',delimiter=',',skiprows=1))
            if len(path)<2 or path.shape[1]!=3 or not np.isfinite(path).all() or np.any(np.linalg.norm(np.diff(path[:,:2],axis=0),axis=1)<=1e-9):
                raise ValueError('invalid reference')
            condition = config['conditions'][trial['task']]
            common_errors, projected, evaluation = tracking_metrics(
                data['t'], np.c_[data['x'], data['y'], data['yaw']], path, condition,
                manifest['starts'][trial['task']]['map_pose'], fresh)
            row.update(common_errors)
            if 'orientation_length_m' in condition:
                write_orientation_series(folder, data, projected, evaluation, fresh, condition, common)
            if fresh.any():
                poses = np.c_[data['x'][fresh],data['y'][fresh],data['yaw'][fresh]]
                e = tracking_errors(poses,path)
                row.update(position_rmse_m=float(np.sqrt(np.mean(e[:,0]**2))),position_max_m=float(e[:,0].max()),
                           yaw_rmse_rad=float(np.sqrt(np.mean(e[:,1]**2))),yaw_max_rad=float(e[:,1].max()))
                row.update(yaw_rmse_deg=math.degrees(row['yaw_rmse_rad']),yaw_max_deg=math.degrees(row['yaw_max_rad']))
                if trial['task']=='E1_lateral':
                    direction = path[-1,:2]-path[0,:2]; direction /= np.linalg.norm(direction)
                    normal = np.array([-direction[1],direction[0]])
                    signed = (poses[:,:2]-path[0,:2])@normal
                    initial = float((np.asarray(manifest['starts'][trial['task']]['map_pose'])[:2]-path[0,:2])@normal)
                    opposite = np.maximum(-np.sign(initial)*signed,0)
                    row.update(lateral_initial_error_m=initial,lateral_crossing_raw_m=float(opposite.max()),
                               lateral_crossing_beyond_deadband_m=float(max(0,opposite.max()-settings['lateral_deadband_m'])))
                    hit = np.flatnonzero(np.abs(signed)<=abs(initial)*settings['lateral_convergence_ratio'])
                    if len(hit) and abs(initial)>1e-9:
                        original_index = np.flatnonzero(fresh)[hit[0]]
                        row['lateral_convergence_time_s']=float(data['t'][original_index])
                        # No distance inferred across missing/stale poses.
                        if fresh[:original_index+1].all():
                            xy = np.vstack((manifest['starts'][trial['task']]['map_pose'][:2],poses[:hit[0]+1,:2]))
                            row['lateral_convergence_distance_m']=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum())
                if trial['task'] in ('E1_orientation_nominal', 'E1_orientation_half'):
                    signed = tracking_errors(poses, path, signed=True)[:, 1]
                    row['yaw_lead_max_rad'] = float(max(0., signed.max()))
                    row['yaw_lag_max_rad'] = float(max(0., -signed.min()))
                    condition = config['conditions'][trial['task']]
                    direction = path[-1, :2] - path[0, :2]
                    direction /= np.linalg.norm(direction)
                    progress = (poses[:, :2] - path[0, :2]) @ direction
                    begin = condition['orientation_start_m']
                    end = begin + condition['orientation_length_m']
                    in_ramp = (progress >= begin - 1e-9) & (progress <= end + 1e-9)
                    slope = abs(condition['goal_yaw_rad']) / condition['orientation_length_m']
                    row['orientation_predicted_speed_m_s'] = common['max_velocity'][2] / slope if slope else None
                    if 'speed_m_s' in data.dtype.names:
                        speeds = data['speed_m_s'][fresh]
                        good = in_ramp & np.isfinite(speeds) & (speeds >= 0)
                        row['orientation_speed_samples'] = int(good.sum())
                        row['orientation_missing_speed_samples'] = int((in_ramp & ~good).sum())
                        if good.any():
                            row['orientation_min_speed_m_s'] = float(speeds[good].min())
                    else:
                        row['orientation_missing_speed_samples'] = int(in_ramp.sum())
                if trial['task']=='E2_environment':
                    fields=('scan_age_s','scan_source_age_s','scan_min_range_m','speed_m_s')
                    if set(fields)<=set(data.dtype.names):
                        sensor_good=fresh & np.isfinite(data['speed_m_s']) & ~np.isnan(data['scan_min_range_m'])
                        for key in fields[:2]:
                            sensor_good &= (data[key]>=0)&(data[key]<=max_age)
                        near=sensor_good & (data['scan_min_range_m']<=settings['obstacle_near_range_m'])
                        row['obstacle_near_samples']=int(near.sum())
                        row['obstacle_near_missing_or_stale_samples']=int((~sensor_good).sum())
                        if near.any():
                            row['obstacle_near_mean_speed_m_s']=float(data['speed_m_s'][near].mean())
                    else:
                        row['obstacle_near_missing_or_stale_samples']=len(data)
                    if config['surveyed_obstacles']:
                        clearance=obstacle_distance(poses[:,:2],config['surveyed_obstacles'])-settings['robot_radius_m']
                        row['surveyed_min_clearance_m']=float(clearance.min())
        except (ValueError,OSError,KeyError) as exc:
            errors.append('reference/condition metrics: '+str(exc))
        row['data_errors']='; '.join(errors)
        diagnostics[trial['id']]=command_metrics(folder,result,common,max_age,settings['maximum_unknown_command_pct'])
        timings[trial['id']]=timing_metrics(folder,result,trial['controller'],diagnostics[trial['id']]['samples'])
        for key in ('constraint_violation_pct','constraint_total_samples','constraint_evaluable_samples',
                    'constraint_violation_samples','constraint_unknown_samples','constraint_unknown_pct',
                    'constraint_violation_lower_pct','constraint_violation_upper_pct','constraint_unknown_flag',
                    'constraint_no_evaluable_cycles','constraint_unknown_threshold_pct'):
            row[key]=diagnostics[trial['id']][key]
        row['timing_samples']=timings[trial['id']]['samples']
        row['timing_status']=timings[trial['id']]['status']
        for stat in ('mean_ms','p95_ms','max_ms'):
            row['compute_time_'+stat]=timings[trial['id']][stat]
        rows.append(row)
    groups=[]
    for task,controller in sorted({(t['task'],t['controller']) for t in manifest['trials']}):
        selected=[r for r in rows if r['task']==task and r['controller']==controller]
        complete=[r for r in selected if r['success'] and r['fresh_pose_samples']>0 and not r['data_errors']
                  and r['invalid_after_warmup_samples']==0 and r['missing_command_prefix_s'] is not None
                  and r['missing_command_prefix_s']<=1/manifest['control_frequency_hz']]
        succeeded=sum(r['success'] for r in selected)
        item=dict(task=task,controller=controller,planned=5,recorded=len(selected),succeeded=succeeded,
                  pending=5-len(selected),failed_or_incomplete=len(selected)-succeeded,valid_successes=len(complete),
                  invalid_or_failed=len(selected)-len(complete),success_rate_attempted=succeeded/len(selected) if selected else None,
                  success_rate_planned=succeeded/5,
                  yaw_error_role='reference_only' if controller in ('RPP','DWPP') else 'tracking',
                  missing_timing_runs=sum(r['timing_samples']==0 for r in selected),
                  missing_constraint_rate_runs=sum(r['constraint_violation_pct'] is None for r in selected),
                  constraint_unknown_samples=sum(r['constraint_unknown_samples'] for r in selected),
                  constraint_unknown_flagged_runs=sum(r['constraint_unknown_flag'] for r in selected),
                  evaluation_complete_count=sum(r['evaluation_complete'] for r in selected))
        numeric_metrics = COMMON_METRICS + transient_metrics(task) + (
            'constraint_unknown_pct','constraint_violation_lower_pct','constraint_violation_upper_pct')
        for metric in numeric_metrics:
            # Retain finite observations from flagged/failed attempts. Travel time
            # is defined only for successful attempts; other missing data stay null.
            values=[r[metric] for r in selected if r[metric] is not None and math.isfinite(r[metric])]
            item[metric+'_n']=len(values)
            item[metric+'_mean']=float(np.mean(values)) if values else None
            item[metric+'_sd']=float(np.std(values,ddof=1)) if len(values)>=2 else None
        groups.append(item)
    if rows:
        with (session/'trial_metrics.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report={'planned':len(manifest['trials']),'recorded':len(rows),'pending':len(manifest['trials'])-len(rows),
            'trials':rows,'groups':groups,'command_diagnostics':diagnostics,'controller_timing':timings,
            'note':'Finite observations from all attempts remain in means, with per-metric n and sample SD. Travel time requires success. Pose errors use fresh poses within the configured spatial window and adjacent-sample trapezoids. Quality flags never discard a trial. Timing covers delegated calls including exceptions, not the server loop. These are observations, not physical-robot validation.'}
    write_group_tables(session, groups)
    (session/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def write_orientation_series(folder, data, projected, evaluation, fresh, condition, common):
    """Plot-ready measured speed, running ramp minimum, and saturation prediction."""
    begin = condition['orientation_start_m']
    end = begin + condition['orientation_length_m']
    slope = abs(condition['goal_yaw_rad']) / condition['orientation_length_m']
    prediction = common['max_velocity'][2]/slope
    minimum = None
    with (folder/'orientation_timeseries.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['t','stamp_s','progress_m','signed_heading_error_deg','in_evaluation',
                         'in_transition','fresh_pose','speed_m_s','transition_min_speed_so_far_m_s','predicted_speed_m_s'])
        for i, sample in enumerate(data):
            ramp = bool(evaluation[i] and begin <= projected[i, 2] <= end)
            speed = float(sample['speed_m_s']) if 'speed_m_s' in data.dtype.names else math.nan
            speed = speed if fresh[i] and math.isfinite(speed) and speed >= 0 else None
            if ramp and speed is not None:
                minimum = speed if minimum is None else min(minimum, speed)
            writer.writerow([sample['t'],sample['stamp_s'],projected[i,2],math.degrees(projected[i,1]),
                             bool(evaluation[i]),ramp,bool(fresh[i]),speed,minimum,prediction])


def write_group_tables(session, groups):
    """One CSV/Markdown table per condition; no transient or settling mixups."""
    destination = session/'tables'
    destination.mkdir(exist_ok=True)
    for task in sorted({g['task'] for g in groups}):
        selected = [g for g in groups if g['task'] == task]
        all_zero = all(g['recorded'] > 0 and g['constraint_violation_pct_n'] == g['recorded']
                       and g['constraint_violation_pct_mean'] == 0 for g in selected)
        metrics = tuple(k for k in COMMON_METRICS if not (all_zero and k == 'constraint_violation_pct')) + transient_metrics(task)
        counts = ('controller','recorded','succeeded','valid_successes','evaluation_complete_count',
                  'constraint_unknown_samples','constraint_unknown_flagged_runs')
        uncertainty = ('constraint_unknown_pct','constraint_violation_lower_pct','constraint_violation_upper_pct')
        columns = list(counts) + [k+'_'+s for k in metrics+uncertainty for s in ('mean','sd','n')]
        with (destination/(task+'.csv')).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
            writer.writeheader(); writer.writerows(selected)
        lines = [f'# {task}', '', 'Values: mean ± sample SD (n). SD is undefined for n < 2.', '',
                 '| Method | Recorded | Success | Quality-qualified | Window complete | Unknown cycles | Flagged runs | ' + ' | '.join(metrics+uncertainty) + ' |',
                 '|' + '---|'*(len(counts)+len(metrics)+len(uncertainty))]
        for g in selected:
            values = [str(g[k]) for k in counts]
            for key in metrics+uncertainty:
                mean, sd, n = (g[key+'_'+s] for s in ('mean','sd','n'))
                values.append(('—' if mean is None else f'{mean:.8g}') + ' ± ' +
                              ('—' if sd is None else f'{sd:.8g}') + f' ({n})')
            lines.append('| ' + ' | '.join(values) + ' |')
        if all_zero:
            note = 'Command constraint violation is 0% for every method over classified cycles; unknown-cycle counts and bounds are reported separately.'
            lines += ['', note]
            (destination/(task+'.note.txt')).write_text(note+'\n')
        else:
            (destination/(task+'.note.txt')).write_text('Constraint percentages use classified cycles; missing rates have zero classified cycles.\n')
        (destination/(task+'.md')).write_text('\n'.join(lines)+'\n')
