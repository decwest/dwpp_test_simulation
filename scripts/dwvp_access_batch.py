#!/usr/bin/env python3
"""Run a frozen schedule; record NavFn/DWVP relocations separately from trials."""
import argparse
from contextlib import contextmanager
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import uuid

import numpy as np
import yaml

import dwvp_access_experiment as experiment
from dwvp_access_path import PathBlockedError, assert_clear, load_map

RETRY_DIRECTION_POLICY = 'retry subset; current stopped poses; turn back along preceding actual path between trials'
CURRENT_POSE_RETRY_DIRECTION_POLICY = 'retry subset; source directions identify original trials; place at each current stopped pose'
LEGACY_RETRY_DIRECTION_POLICY = 'retry subset; preserve source directions; unscored alignment before each leg'
TURNAROUND_CLEARANCE_M = .03
TURNAROUND_MAX_OFFSET_DEG = 5
TURNAROUND_MAX_ATTEMPTS = 3
RETRY_RESUME_RELOCATION_DISTANCE_M = .5


def settled_endpoint_failure(result, manifest, folder):
    """Only a measured endpoint miss after successful action and verified stop.

    Older records have no failure_reason; require their final fresh, stationary
    CSV sample as well. A controller abort, stale data or failed stop never qualifies.
    """
    if (result.get('status') != 'failed' or result.get('success') is not False
            or result.get('action_status') != 4
            or result.get('endpoint_policy') != 'fixed_tolerance'
            or result.get('final_pose_within_tolerances') is not False
            or result.get('settling', {}).get('verified') is not True):
        return False
    if result.get('failure_reason') not in (None, 'endpoint_tolerance_exceeded'):
        return False
    if not result.get('error', '').startswith('Final stopped pose outside tolerances or action unsuccessful:'):
        return False
    try:
        pose = np.asarray(result['final_pose'], dtype=float)
        xy, yaw = result['final_position_error_m'], result['final_yaw_error_rad']
        if (pose.shape != (3,) or not np.isfinite(pose).all()
                or not all(math.isfinite(v) and v >= 0 for v in (xy, yaw))
                or not (xy > manifest['xy_tolerance_m'] or yaw > manifest['yaw_tolerance_rad'])):
            return False
        if 'final_pose_fresh' in result:
            return result['final_pose_fresh'] is True
        with (folder/'settling.csv').open() as stream:
            last = list(csv.DictReader(stream))[-1]
        return (last['fresh'] == 'True' and last['applied_zero'] == 'True'
                and 0 <= float(last['speed_m_s']) <= .005
                and 0 <= float(last['yaw_rate_rad_s']) <= .01
                and np.allclose(pose, [float(last[k]) for k in ('x', 'y', 'yaw')], atol=1e-6, rtol=0))
    except (OSError, KeyError, IndexError, TypeError, ValueError):
        return False


def validate_retry_manifest(session, manifest):
    """A retry may contain consecutive reverse legs, but may not change a leg."""
    provenance = manifest['retry_of']
    source_file = Path(session)/'retry_source_manifest.json'
    results_file = Path(session)/'retry_source_results.json'
    if (experiment.digest(source_file) != provenance['manifest_sha256']
            or experiment.digest(results_file) != provenance['results_sha256']):
        raise ValueError('Retry source snapshot changed')
    original = json.loads(source_file.read_text())
    results = json.loads(results_file.read_text())
    expected = [t for t in original['trials'] if t['id'] in results]
    if not expected or len(expected) != len(results) or manifest['trials'] != expected:
        raise ValueError('Retry must preserve the selected source trials, order and directions')
    selection = provenance.get('selection', 'failed_trials')
    if selection == 'explicit_trial':
        if len(expected) != 1 or expected[0]['id'] != provenance.get('trial_id'):
            raise ValueError('Explicit retry must contain exactly the requested trial')
        result = results[expected[0]['id']]['result']
        if not (verified_success(result) or (result.get('status') == 'failed' and result.get('success') is False
                and result.get('action_status') == 4 and result.get('settling', {}).get('verified') is True)):
            raise ValueError('Explicit retry requires a completed, stopped source trial')
    elif selection == 'failed_trials':
        if any(r['result'].get('status') != 'failed' or r['result'].get('success') is not False
               for r in results.values()):
            raise ValueError('Retry must preserve the failed source trials, order and directions')
    else:
        raise ValueError('Unknown retry selection policy')
    if manifest.get('direction_policy') not in (RETRY_DIRECTION_POLICY, LEGACY_RETRY_DIRECTION_POLICY,
                                               CURRENT_POSE_RETRY_DIRECTION_POLICY):
        raise ValueError('Unknown retry direction policy')
    wanted = dict(original, trials=expected, retry_of=provenance,
                  direction_policy=manifest['direction_policy'])
    if manifest != wanted:
        raise ValueError('Retry changed frozen experimental conditions')


def failed_trials(session):
    """Select completed endpoint misses, after validating the whole source run."""
    manifest, pending = selected_trials(session, resume=True, continue_on_endpoint_failure=True)
    if pending:
        raise ValueError('Finish the pending trials with --resume before --retry-failed')
    failed = [t for t in manifest['trials']
              if json.loads((Path(session)/'runs'/t['id']/'result.json').read_text())['success'] is False]
    return manifest, failed


def prepare_failed_retry(source, output):
    """Freeze just failed trials in a new session, retaining the original evidence."""
    return prepare_retry(source, output)


def verified_success(result):
    return (result.get('status') == 'succeeded' and result.get('success') is True
            and result.get('action_status') == 4 and result.get('settling', {}).get('verified') is True)


def explicit_retry_trial(session, trial_id):
    """Select one completed trial for operator-requested reacquisition, including successes."""
    manifest, pending = selected_trials(session, resume=True, continue_on_endpoint_failure=True)
    if pending:
        raise ValueError('Finish the pending trials with --resume before selecting a new retry')
    trial = next((t for t in manifest['trials'] if t['id'] == trial_id), None)
    if trial is None:
        raise ValueError(f'Unknown trial in source session: {trial_id}')
    folder = Path(session)/'runs'/trial_id
    result = json.loads((folder/'result.json').read_text())
    if not (verified_success(result) or settled_endpoint_failure(result, manifest, folder)):
        raise ValueError('Explicit retry requires a completed, stopped source trial')
    return manifest, [trial]


def prepare_retry(source, output, *, trial_id=None):
    """Freeze failures or one explicitly selected trial; never rewrite source outcomes."""
    source, output = Path(source).resolve(), Path(output)
    with (source/'.batch.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another batch owns the source session')
        manifest, trials = (failed_trials(source) if trial_id is None else explicit_retry_trial(source, trial_id))
        if not trials:
            raise ValueError('No failed trials to retry')
        frozen = {manifest['params_file']: manifest['params_sha256'],
                  manifest['config_file']: manifest['config_sha256']}
        for item in [*manifest['parameter_sets'].values(), *manifest['paths'].values()]:
            if item.get('file'):
                frozen[item['file']] = item['sha256']
        # Read and verify everything before reserving the destination.
        contents = {}
        for name, expected in frozen.items():
            path = source/name
            if Path(name).is_absolute() or '..' in Path(name).parts or experiment.digest(path) != expected:
                raise ValueError('Frozen retry input changed or has an invalid path')
            contents[name] = path.read_bytes()
        source_manifest = (source/'manifest.json').read_bytes()
        results = {t['id']: dict(result=json.loads((source/'runs'/t['id']/'result.json').read_text()),
                    result_sha256=experiment.digest(source/'runs'/t['id']/'result.json')) for t in trials}
        output.mkdir(parents=True, exist_ok=False)
        for name, data in contents.items():
            path = output/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
        for name in ('params_template.yaml', 'start_capture.json'):
            if (source/name).is_file():
                shutil.copy2(source/name, output/name)
        (output/'retry_source_manifest.json').write_bytes(source_manifest)
        experiment.write_json(output/'retry_source_results.json', results)
        provenance = dict(session=str(source), manifest_sha256=experiment.digest(output/'retry_source_manifest.json'),
                          results_sha256=experiment.digest(output/'retry_source_results.json'))
        if trial_id is not None:
            provenance.update(selection='explicit_trial', trial_id=trial_id,
                              reason='operator_requested_reacquisition')
        retry = dict(manifest, trials=trials, retry_of=provenance,
                     direction_policy=RETRY_DIRECTION_POLICY)
        experiment.write_json(output/'manifest.json', retry)
        validate_retry_manifest(output, retry)
        return retry


def selected_trials(session, conditions=None, repeats=None, resume=False, continue_on_endpoint_failure=False):
    manifest = json.loads((Path(session) / 'manifest.json').read_text())
    if 'retry_of' in manifest:
        validate_retry_manifest(session, manifest)
    requested = conditions or list(experiment.CONDITIONS)
    requested = [c for item in requested for c in
                 (experiment.CONDITIONS[:-1] if item == 'E1' else [item])]
    if not set(requested) <= set(experiment.CONDITIONS):
        raise ValueError('Unknown condition selection')
    if repeats is not None and (not repeats or not set(repeats) <= set(range(1, 6))):
        raise ValueError('Repeats must be in 1..5')
    trials = [t for t in manifest['trials'] if t['task'] in requested
              and (repeats is None or t['repeat'] in repeats)]
    if not trials:
        raise ValueError('Empty schedule')
    if 'retry_of' not in manifest and manifest.get('bidirectional') and any(t.get('direction') != ('forward' if i % 2 == 0 else 'reverse')
                                           for i, t in enumerate(trials)):
        raise ValueError('Selection breaks frozen forward/reverse alternation; prepare a new session with --conditions/--repeats')
    completed = 0
    pending_seen = False
    for trial in trials:
        try:
            experiment.load_trial(session, trial['id'])
        except ValueError as exc:
            raise ValueError(f"{trial['id']}: {exc}. Supply the E2 route or explicitly select --conditions E1") from exc
        folder = Path(session) / 'runs' / trial['id']
        if not folder.exists():
            pending_seen = True
            continue
        if not resume:
            raise FileExistsError(f"Trial already reserved/recorded: {trial['id']}; use --resume for a successful prefix")
        if pending_seen:
            raise ValueError('Resume requires completed trials to be a contiguous prefix of the selected schedule')
        try:
            result = json.loads((folder/'result.json').read_text())
            recorded = json.loads((folder/'trial.json').read_text())
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot resume incomplete trial {trial['id']}; existing attempts are never retried") from exc
        succeeded = result.get('success') is True and result.get('status') == 'succeeded'
        endpoint_miss = continue_on_endpoint_failure and settled_endpoint_failure(result, manifest, folder)
        if not succeeded and not endpoint_miss:
            raise ValueError(f"Cannot resume failed trial {trial['id']}; existing attempts are never retried")
        if (recorded.get('trial') != trial
                or recorded.get('manifest_sha256') != experiment.digest(Path(session)/'manifest.json')
                or recorded.get('params_sha256') != trial['params_sha256']):
            raise ValueError(f"Recorded inputs differ from frozen session for {trial['id']}")
        if recorded.get('reference_policy') == 'per_trial_current_pose':
            experiment.recorded_geometry(session, trial, folder)
        completed += 1
    return manifest, trials[completed:]


def check_geometry(session, trials, map_file):
    info, pixels, blocked = load_map(map_file)
    checked = set()
    for trial in trials:
        geometry_key = (trial['task'], trial.get('direction', 'forward'))
        if geometry_key in checked:
            continue
        manifest, _, path = experiment.load_trial(session, trial['id'])
        initial = experiment.start_pose(manifest, trial)
        check_placed_geometry(path, initial, Path(session)/trial['params_file'], (info, pixels, blocked))
        checked.add(geometry_key)
    return info, pixels, blocked


def check_placed_geometry(path, initial, params_file, map_data, *, clearance=0.):
    info, _, blocked = map_data
    params = yaml.safe_load(Path(params_file).read_text())
    radius = params['local_costmap']['local_costmap']['ros__parameters']['robot_radius']
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError('Positive circular robot radius required')
    approach = np.vstack((initial, path)) if np.linalg.norm(initial[:2]-path[0, :2]) > 1e-6 else path
    if not math.isfinite(clearance) or clearance < 0:
        raise ValueError('Nonnegative finite placement clearance required')
    assert_clear(approach, info, blocked, radius+clearance)


def turnaround_yaw(previous_reference, next_reference, next_start):
    """Reverse path travel, independently of the preceding trial's final body yaw."""
    axes = []
    for path in (previous_reference, next_reference):
        delta = np.asarray(path)[-1, :2]-np.asarray(path)[0, :2]
        if not np.isfinite(delta).all() or np.linalg.norm(delta) < 1e-6:
            raise ValueError('Turnaround requires a nonzero finite travel direction')
        axes.append(math.atan2(delta[1], delta[0]))
    return float(experiment.wrap(axes[0]+math.pi-(axes[1]-next_start[2])))


def clear_turnaround_target(manifest, trial, reference, params, map_data, current, nominal_yaw):
    """Keep the current XY and local trial shape; select a nearby return heading.

    A full-length return can extend beyond the preceding stopped endpoint. Small
    localization shifts can also make the exactly antiparallel path intersect a
    wall. Search only +/-5 degrees, nearest first, with extra map clearance.
    """
    candidates = [0, *(offset for step in range(1, TURNAROUND_MAX_OFFSET_DEG+1) for offset in (-step, step))]
    checks = []
    for offset in candidates:
        target = np.array([*current[:2], float(experiment.wrap(nominal_yaw+math.radians(offset)))])
        _, path = experiment.reanchor_trial(manifest, trial, reference, target)
        check = dict(offset_deg=offset, target=target.tolist(), clear=False)
        checks.append(check)
        try:
            check_placed_geometry(path, target, params, map_data, clearance=TURNAROUND_CLEARANCE_M)
        except PathBlockedError as error:
            check['collision'] = error.details
        else:
            check['clear'] = True
            return target, checks
    return None, checks


def checked_retry_turnaround(session, manifest, trial, reference, params, map_data,
                             capture_stop, turn, folder, *, resuming_first=False):
    """First retry uses the operator's pose; subsequent ones reverse actual travel.

    Resolve the nominal direction from recorded geometry even on resume. Check
    the *settled* path, since action success need not mean the final heading was
    reached. Bounded in-place corrections never change the frozen local trial.
    On resume, an operator may have moved to a new start. Only the first pending
    leg may then use the operator's current heading instead of the old return.
    """
    index = manifest['trials'].index(trial)
    if index == 0:
        return None
    previous = manifest['trials'][index-1]
    previous_folder = Path(session)/'runs'/previous['id']
    result = json.loads((previous_folder/'result.json').read_text())
    if not (result.get('status') == 'succeeded' and result.get('success') is True
            or settled_endpoint_failure(result, manifest, previous_folder)):
        raise ValueError('Cannot turn around after an incomplete or unsafe preceding retry')
    _, previous_path, _ = experiment.recorded_geometry(session, previous, previous_folder)
    captured = capture_stop()
    nominal_yaw = turnaround_yaw(previous_path, reference, experiment.start_pose(manifest, trial))
    folder.mkdir(parents=True, exist_ok=True)
    record = dict(previous_trial=previous['id'],
        previous_reference_sha256=experiment.digest(previous_folder/'reference.csv'),
        before=captured, nominal_yaw=nominal_yaw, clearance_m=TURNAROUND_CLEARANCE_M,
        max_heading_offset_deg=TURNAROUND_MAX_OFFSET_DEG, max_attempts=TURNAROUND_MAX_ATTEMPTS,
        attempts=[], static_map_checked=False, stopped_heading_verified=False, stopped_path_checked=False)
    record['start_mode'] = 'previous_trial_return'
    record['resuming_first'] = resuming_first
    record_file = folder/(trial['id']+'.json')
    experiment.write_json(record_file, record)
    if resuming_first:
        previous_stop = np.asarray(result.get('final_pose'), dtype=float)
        if previous_stop.shape != (3,) or not np.isfinite(previous_stop).all():
            raise ValueError('Cannot choose retry resume direction: previous final pose is missing or invalid')
        distance = float(np.linalg.norm(np.asarray(captured['pose'])[:2]-previous_stop[:2]))
        record.update(previous_stopped_pose=previous_stop.tolist(), distance_from_previous_stop_m=distance,
                      resume_relocation_distance_m=RETRY_RESUME_RELOCATION_DISTANCE_M)
        experiment.write_json(record_file, record)
        if distance > RETRY_RESUME_RELOCATION_DISTANCE_M:
            # Do not impose a historical map direction after the operator has
            # repositioned. Keep the full reference and clearance checks.
            _, actual_path = experiment.reanchor_trial(manifest, trial, reference, captured['pose'])
            path_file = folder/(trial['id']+'_relocated_resume.csv')
            np.savetxt(path_file, actual_path, delimiter=',', header='x,y,yaw', comments='')
            record.update(start_mode='relocated_current_pose', target=captured['pose'], after=captured,
                          turn_required=False, path_file=path_file.name, stopped_yaw_error_rad=0.,
                          stopped_heading_verified=True)
            try:
                check_placed_geometry(actual_path, captured['pose'], params, map_data,
                                      clearance=TURNAROUND_CLEARANCE_M)
            except PathBlockedError as collision:
                record['collision'] = collision.details
                experiment.write_json(record_file, record)
                raise RuntimeError(f'Path from the repositioned current pose is blocked; no trial goal sent. '
                                   f'Choose an open start position/direction and resume. See {record_file}') from collision
            record.update(static_map_checked=True, stopped_path_checked=True)
            experiment.write_json(record_file, record)
            print(f'  Repositioned resume ({distance:.2f} m from previous stop): '
                  'use the current position and heading for this leg.', flush=True)
            return record
    for attempt_index in range(TURNAROUND_MAX_ATTEMPTS):
        target, candidates = clear_turnaround_target(manifest, trial, reference, params, map_data,
                                                     captured['pose'], nominal_yaw)
        attempt = dict(number=attempt_index+1, before=captured, candidates=candidates)
        record['attempts'].append(attempt)
        if target is None:
            record['collision'] = candidates[0]['collision']
            record['static_map_checked'] = False
            experiment.write_json(record_file, record)
            raise RuntimeError(f'No clear return-direction trial path within +/-{TURNAROUND_MAX_OFFSET_DEG} deg; '
                               f'no turnaround goal sent. Reposition and resume. See {record_file}')
        record.update(target=target.tolist(), static_map_checked=True)
        experiment.write_json(record_file, record)
        turn(target)
        stopped = capture_stop()
        _, actual_path = experiment.reanchor_trial(manifest, trial, reference, stopped['pose'])
        path_file = folder/(trial['id']+f'_attempt_{attempt_index+1:02d}.csv')
        np.savetxt(path_file, actual_path, delimiter=',', header='x,y,yaw', comments='')
        error = abs(float(experiment.wrap(stopped['pose'][2]-target[2])))
        attempt.update(after=stopped, target=target.tolist(), stopped_yaw_error_rad=error,
                       stopped_heading_verified=error <= .05, stopped_path_checked=False, path_file=path_file.name)
        try:
            check_placed_geometry(actual_path, stopped['pose'], params, map_data, clearance=TURNAROUND_CLEARANCE_M)
        except PathBlockedError as collision:
            attempt['collision'] = collision.details
        else:
            attempt['stopped_path_checked'] = True
        record.update({key: attempt[key] for key in (
            'after', 'stopped_yaw_error_rad', 'stopped_heading_verified', 'stopped_path_checked')})
        experiment.write_json(record_file, record)
        if record['stopped_heading_verified'] and record['stopped_path_checked']:
            return record
        if attempt_index+1 < TURNAROUND_MAX_ATTEMPTS:
            print('  Settled return path/heading needs an in-place correction; '
                  f'attempt {attempt_index+2}/{TURNAROUND_MAX_ATTEMPTS}', flush=True)
            captured = capture_stop()
    raise RuntimeError(f'Turnaround did not produce a clear settled path after {TURNAROUND_MAX_ATTEMPTS} attempts; '
                       f'no trial goal sent. Reposition and resume. See {record_file}')


def checked_current_placement(manifest, trial, reference, params, map_data,
                              capture_stop, align_start, folder, *, current_only=False):
    """Retries start where the robot stopped; ordinary batches may align first."""
    folder.mkdir(parents=True, exist_ok=True)
    target = experiment.start_pose(manifest, trial)
    if not current_only:
        check_placed_geometry(reference, target, params, map_data)
        align_start(target)
    captured = capture_stop()
    placed, path = experiment.reanchor_trial(manifest, trial, reference, captured['pose'])
    path_file = folder/(trial['id']+'.csv')
    np.savetxt(path_file, path, delimiter=',', header='x,y,yaw', comments='')
    diagnostic = dict(trial_id=trial['id'], capture=captured,
        alignment_target=None if current_only else target.tolist(),
        start_policy='current_pose_with_turnaround' if current_only else 'align_then_current_pose',
        source_direction=trial.get('direction','forward'), path_file=path_file.name, static_map_checked=False)
    try:
        check_placed_geometry(path, captured['pose'], params, map_data)
    except PathBlockedError as error:
        diagnostic['collision'] = error.details
        experiment.write_json(folder/(trial['id']+'.json'), diagnostic)
        raise RuntimeError(f'Path from the current pose is blocked; no trial goal sent. '
                           f'Choose an open start position/direction and resume. '
                           f'Inspect {folder/(trial["id"]+".json")}') from error
    diagnostic['static_map_checked'] = True
    experiment.write_json(folder/(trial['id']+'.json'), diagnostic)
    return captured, placed, path


def map_matches(msg, info, pixels):
    if msg is None or msg.header.frame_id != 'map':
        return False
    h, w = pixels.shape
    q = msg.info.origin.orientation
    angle = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
    if (msg.info.width != w or msg.info.height != h
            or not np.allclose([msg.info.resolution, msg.info.origin.position.x,
                                msg.info.origin.position.y, angle],
                               [info['resolution'], *info['origin']], atol=1e-6, rtol=0)):
        return False
    occ = pixels / 255. if info.get('negate', 0) else (255-pixels) / 255.
    expected = np.where(occ > info['occupied_thresh'], 100,
                        np.where(occ < info['free_thresh'], 0, -1))
    return np.array_equal(np.asarray(msg.data), np.flipud(expected).ravel())


def relocation_path(xy, current, target):
    """Keep planner geometry; interpolate body yaw independently for omni motion."""
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError('Planner returned invalid positions')
    xy = np.vstack((np.asarray(current)[:2], xy, np.asarray(target)[:2]))
    xy = xy[np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-6]]
    if len(xy) < 2:
        return np.array([current, target], dtype=float)
    arc = experiment.arclength(xy)
    yaw = experiment.wrap(current[2] + arc/arc[-1] * float(experiment.wrap(target[2]-current[2])))
    return experiment.validate_path(np.c_[xy, yaw])


def pose_close(current, target, manifest):
    return (np.linalg.norm(np.asarray(current)[:2]-np.asarray(target)[:2]) <= manifest['xy_tolerance_m']
            and abs(float(experiment.wrap(current[2]-target[2]))) <= manifest['yaw_tolerance_rad'])


def stop_process(process):
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def spin_until(observer, predicate, timeout, guard=None):
    import rclpy
    end = time.monotonic() + timeout
    while not predicate():
        if getattr(observer, 'cancel_requested', False):
            raise KeyboardInterrupt('Batch interrupted')
        if time.monotonic() >= end:
            raise TimeoutError('Timed out waiting for ROS/process operation')
        rclpy.spin_once(observer, timeout_sec=.02)
        if guard:
            guard()


def service(observer, kind, name, request):
    client = observer.create_client(kind, name)
    try:
        spin_until(observer, client.service_is_ready, 15)
        future = client.call_async(request)
        spin_until(observer, future.done, 30)
        return future.result()
    finally:
        observer.destroy_client(client)


@contextmanager
def controller_stack(observer, params_file, folder, *, turnaround=False):
    """Own only controller, smoother and return planner; leave AMCL running."""
    from ament_index_python.packages import get_package_prefix
    from lifecycle_msgs.srv import ChangeState
    names = {'controller_server', 'velocity_smoother', 'planner_server'}
    spin_until(observer, lambda: not any(name in names for name, _ in observer.get_node_names_and_namespaces()), 10)
    observer.command_owners(False)
    params = yaml.safe_load(Path(params_file).read_text())
    if turnaround:
        params = experiment.with_turnaround_checker(params)
    params['planner_server'] = {'ros__parameters': {
        'planner_plugins': ['NavFn'], 'expected_planner_frequency': 1.,
        'NavFn': {'plugin': 'nav2_navfn_planner/NavfnPlanner', 'tolerance': .05,
                  'use_astar': False, 'allow_unknown': False}}}
    folder.mkdir(parents=True)
    combined = folder / 'stack_params.yaml'
    combined.write_text(yaml.safe_dump(params, sort_keys=False))
    processes, logs = [], []
    try:
        for package, binary, remaps in (
            ('nav2_controller', 'controller_server', ['cmd_vel:=/cmd_vel_nav']),
            ('nav2_velocity_smoother', 'velocity_smoother',
             ['cmd_vel:=/cmd_vel_nav', 'cmd_vel_smoothed:=/omni_base_controller/cmd_vel']),
            ('nav2_planner', 'planner_server', []),
        ):
            log = (folder / (binary + '.log')).open('w'); logs.append(log)
            command = [str(Path(get_package_prefix(package)) / 'lib' / package / binary),
                       '--ros-args', '--params-file', str(combined)]
            for remap in remaps:
                command.extend(['-r', remap])
            processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True))
        for name in ('controller_server', 'velocity_smoother', 'planner_server'):
            for transition in (1, 3):
                request = ChangeState.Request(); request.transition.id = transition
                if not service(observer, ChangeState, '/' + name + '/change_state', request).success:
                    raise RuntimeError(f'{name} rejected lifecycle transition {transition}')
        observer.stopped()
        observer.command_owners(True)
        yield processes
    finally:
        # Let the smoother finish deceleration after the controller has stopped.
        # This also covers cancellation before its final zero reaches the base.
        try:
            if processes:
                stop_process(processes[0])
            if len(processes) >= 2 and processes[1].poll() is None:
                smooth = params['velocity_smoother']['ros__parameters']
                bound = max(v / abs(a) for v, a in zip(smooth['max_velocity'], smooth['max_decel']))
                stopped = observer.await_zero_output(bound + smooth.get('velocity_timeout', .5) + 2.)
                experiment.write_json(folder / 'shutdown.json', stopped)
        finally:
            for process in processes:
                stop_process(process)
            for log in logs:
                log.close()


def planned_return(observer, current, target):
    if np.linalg.norm(np.asarray(current)[:2]-np.asarray(target)[:2]) <= 1e-6:
        return np.array([current, target], dtype=float)
    from action_msgs.msg import GoalStatus
    from nav2_msgs.action import ComputePathToPose
    from rclpy.action import ActionClient
    client = ActionClient(observer, ComputePathToPose, '/compute_path_to_pose')
    try:
        spin_until(observer, client.server_is_ready, 15)
        goal = ComputePathToPose.Goal()
        poses = experiment.pose_path_message([current, target], 'map', observer.get_clock().now().to_msg()).poses
        goal.start, goal.goal = poses
        goal.use_start = True; goal.planner_id = 'NavFn'
        future = client.send_goal_async(goal)
        spin_until(observer, future.done, 10)
        handle = future.result()
        if not handle.accepted:
            raise RuntimeError('NavFn rejected return planning')
        future = handle.get_result_async()
        spin_until(observer, future.done, 30)
        response = future.result()
        if response.status != GoalStatus.STATUS_SUCCEEDED or response.result.path.header.frame_id != 'map':
            raise RuntimeError('NavFn could not find a return path in map coordinates')
        return relocation_path([[p.pose.position.x, p.pose.position.y] for p in response.result.path.poses], current, target)
    finally:
        client.destroy()


def recorded_child(observer, command, logfile, expected_result, processes, map_data,
                   *, endpoint_manifest=None):
    """Continuously observe the robot while the existing recorder owns the goal."""
    invalid_since = None
    def guard():
        nonlocal invalid_since
        if any(p.poll() is not None for p in processes):
            raise RuntimeError('An owned navigation process exited')
        try:
            observer.state()
            observer.command_owners(True)
            if not map_matches(observer.map, *map_data):
                raise RuntimeError('Live /map differs from the checked map')
            invalid_since = None
        except Exception:
            if invalid_since is None:
                invalid_since = time.monotonic()
            if time.monotonic() - invalid_since > .3:
                raise
    with logfile.open('w') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            spin_until(observer, lambda: child.poll() is not None, 190, guard)
        finally:
            stop_process(child)  # SIGINT lets the recorder cancel its goal and save results.
    try:
        result = json.loads(expected_result.read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError(f'Recorder exited {child.returncode} without a readable result; see {logfile}') from exc
    endpoint_miss = (child.returncode == 0 and endpoint_manifest is not None
                     and settled_endpoint_failure(result, endpoint_manifest, expected_result.parent))
    if child.returncode or (result.get('success') is not True and not endpoint_miss):
        raise RuntimeError(f'Recorder failed (exit {child.returncode}, status {result.get("status")}): '
                           f'{result.get("error", "Inspect final errors in " + str(expected_result))}; see {logfile}')
    observer.stopped()
    if any(p.poll() is not None for p in processes):
        raise RuntimeError('An owned navigation process exited')
    observer.command_owners(True)
    if not map_matches(observer.map, *map_data):
        raise RuntimeError('Live /map differs from the checked map')
    if endpoint_miss:
        print(f'  Endpoint miss recorded as FAILED; continuing after verified stop: '
              f'{result["error"]}', flush=True)
    return result


def execute(args, manifest, trials, map_data, output):
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Path as RosPath
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from rclpy.signals import SignalHandlerOptions
    from dwvp_access_runtime import Observer
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    observer = Observer(args.base_frame, args.odom_topic)
    observer.cancel_requested = False
    def request_stop(signum, frame):
        # Defer Python exceptions until outside rclpy's C message conversion.
        observer.cancel_requested = True
    previous_signals = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    visual_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    reference_pub = observer.create_publisher(RosPath, '/dwvp_access/reference_path', visual_qos)
    start_pub = observer.create_publisher(PoseStamped, '/dwvp_access/start_pose', visual_qos)
    return_pub = observer.create_publisher(RosPath, '/dwvp_access/return_path', visual_qos)
    session = args.session.resolve()
    status = dict(status='starting', completed=[], failed_trials=[], returns=[], alignments=[], turnarounds=[], motion_requested=False)
    status['continue_on_endpoint_failure'] = bool(args.continue_on_endpoint_failure)
    status['resume'] = bool(args.resume)
    status['reference_policy'] = 'per_trial_current_pose' if args.start_from_current else 'session_fixed'
    current_only = 'retry_of' in manifest and args.start_from_current
    status['start_policy'] = 'current_pose_with_turnaround' if current_only else 'align_then_current_pose'
    experiment.write_json(output / 'schedule.json', trials)
    info, pixels, blocked = map_data

    def align_to_start(target, trial, processes, params, kind, record_id, *, reanchor_after_stop=False,
                       heading_only=False):
        current = observer.stopped()['pose']
        config = yaml.safe_load(params.read_text())
        if heading_only:
            target = np.array([*current[:2], target[2]])
            close = abs(experiment.wrap(current[2]-target[2])) <= experiment.TURNAROUND_YAW_TOLERANCE
        else:
            close = pose_close(current, target, manifest)
        if close:
            return
        print(f'  {kind} -> {target.tolist()}', flush=True)
        route = np.array([current, target]) if heading_only else planned_return(observer, current, target)
        radius = config['local_costmap']['local_costmap']['ros__parameters']['robot_radius']
        assert_clear(route, info, blocked, radius)
        return_pub.publish(experiment.pose_path_message(route, 'map', observer.get_clock().now().to_msg()))
        packet = dict(path=route.tolist(), start=current, output=str(output/kind/record_id),
                      endpoint_policy='reanchor_after_stop' if reanchor_after_stop else 'fixed_tolerance')
        if heading_only:
            packet['goal_checker_id'] = experiment.TURNAROUND_GOAL_CHECKER
        packet_file = output/(record_id+'_return.json')
        experiment.write_json(packet_file, packet)
        status.update(status=kind, motion_requested=True)
        experiment.write_json(output/'status.json', status)
        recorded_child(observer, [sys.executable, str(Path(__file__)), '_transfer',
            '--session', str(session), '--trial', trial['id'], '--packet', str(packet_file),
            '--base-frame', args.base_frame, '--odom-topic', args.odom_topic],
            output/(record_id+'_return.log'), Path(packet['output'])/'result.json', processes, (info, pixels))
        if not reanchor_after_stop and not pose_close(observer.stopped()['pose'], target, manifest):
            raise RuntimeError('Positioning did not reach the next frozen start')
        status[kind].append(record_id)

    try:
        initial = observer.stopped()
        observer.command_owners(False)
        if not map_matches(observer.map, info, pixels):
            raise RuntimeError('Live /map differs from --map; launch localization with the checked map')
        if not args.resume and not args.start_from_current:
            experiment.check_start_pose(initial['pose'], manifest, trials[0])
        if observer.get_publishers_info_by_topic('/global_costmap/costmap'):
            raise RuntimeError('Stop existing navigation/planner processes before batch execution')
        index = 0
        while index < len(trials):
            trial = trials[index]
            params = session / trial['params_file']
            with controller_stack(observer, params, output / f'stack_{index:03d}', turnaround=current_only) as processes:
                if args.resume and index == 0 and not args.start_from_current:
                    # The previous batch may have stopped during an unscored
                    # alignment. Preserve its files and record this new transfer.
                    align_to_start(experiment.start_pose(manifest, trial), trial, processes, params,
                                   'alignments', 'resume_'+trial['id'])
                # Restart the owned stack only when frozen parameters change.
                while index < len(trials) and trials[index]['params_sha256'] == experiment.digest(params):
                    trial = trials[index]
                    _, _, reference = experiment.load_trial(session, trial['id'])
                    placed_trial = trial
                    capture_args = []
                    if args.start_from_current:
                        target = experiment.start_pose(manifest, trial)
                        turnaround = None
                        if current_only:
                            turn_attempt = 0
                            def turn(pose):
                                nonlocal turn_attempt
                                turn_attempt += 1
                                align_to_start(pose, trial, processes, params, 'turnarounds',
                                               f'before_{trial["id"]}_attempt_{turn_attempt:02d}',
                                               reanchor_after_stop=True, heading_only=True)
                            turnaround = checked_retry_turnaround(session, manifest, trial, reference, params,
                                map_data, observer.stopped, turn, output/'turnaround_checks',
                                resuming_first=args.resume and index == 0)
                        def align_start(pose):
                            align_to_start(pose, trial, processes, params, 'alignments', 'start_'+trial['id'],
                                           reanchor_after_stop=True)
                        # Place using the very same fresh stopped capture that
                        # passed the turnaround map check, including its margin.
                        capture_stop = (lambda: turnaround['after']) if turnaround is not None else observer.stopped
                        captured, placed_trial, reference = checked_current_placement(
                            manifest, trial, reference, params, map_data, capture_stop, align_start,
                            output/'path_checks', current_only=current_only)
                        capture_file = output/(trial['id']+'_start_capture.json')
                        experiment.write_json(capture_file, dict(trial_id=trial['id'], capture=captured,
                            manifest_sha256=experiment.digest(session/'manifest.json'),
                            policy='per_trial_current_pose', start_policy=status['start_policy'],
                            alignment_target=None if current_only else target.tolist(),
                            turnaround=turnaround,
                            static_map_checked=True))
                        capture_args = ['--start-capture', str(capture_file)]
                        print(f"  measured start -> {captured['pose']}", flush=True)
                    experiment.check_start_pose(observer.stopped()['pose'], manifest, placed_trial)
                    observer.command_owners(True)
                    stamp = observer.get_clock().now().to_msg()
                    reference_pub.publish(experiment.pose_path_message(reference, 'map', stamp))
                    start_pub.publish(experiment.pose_path_message(
                        [experiment.start_pose(manifest, placed_trial)], 'map', stamp).poses[0])
                    empty = RosPath(); empty.header.frame_id = 'map'; return_pub.publish(empty)
                    label = 'current-pose retry' if current_only else trial.get('direction', 'forward')+' trial'
                    print(f"[{index+1}/{len(trials)}] {label} {trial['id']}", flush=True)
                    status.update(status='trial', current=trial['id'], motion_requested=True)
                    experiment.write_json(output / 'status.json', status)
                    trial_result = recorded_child(observer, [sys.executable, str(Path(experiment.__file__)), 'run',
                        '--session', str(session), '--trial', trial['id'], '--base-frame', args.base_frame,
                        '--odom-topic', args.odom_topic] + capture_args, output / (trial['id'] + '.log'),
                        session / 'runs' / trial['id'] / 'result.json', processes, (info, pixels),
                        endpoint_manifest=manifest if args.continue_on_endpoint_failure else None)
                    status['completed'].append(trial['id'])
                    if not trial_result['success']:
                        status['failed_trials'].append(trial['id'])
                    if args.start_from_current:
                        index += 1
                        experiment.write_json(output/'status.json', status)
                        continue
                    if manifest.get('bidirectional') and index + 1 == len(trials):
                        # Each traverse is scored. Do not add an extra return
                        # after the last leg or silently create a 56th/81st trial.
                        index += 1
                        break
                    # All current-start E1 trials share this target. E2 can have
                    # a different map-fixed start; the last leg returns home.
                    target = experiment.start_pose(manifest, trials[(index+1) % len(trials)])
                    align_to_start(target, trial, processes, params,
                                   'alignments' if manifest.get('bidirectional') else 'returns', trial['id'])
                    index += 1
                    experiment.write_json(output / 'status.json', status)
        status['status'] = 'completed'
    except (Exception, KeyboardInterrupt) as exc:
        status.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc))
        raise
    finally:
        experiment.write_json(output / 'status.json', status)
        observer.destroy_node()
        rclpy.try_shutdown()
        experiment.summarize(session)
        for sig, handler in previous_signals.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    transfer = len(sys.argv) > 1 and sys.argv[1] == '_transfer'
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--base-frame', default='base_link')
    parser.add_argument('--odom-topic', default='/omni_base_controller/wheel_odom')
    if transfer:
        parser.add_argument('--trial', required=True)
        parser.add_argument('--packet', type=Path, required=True)
        args = parser.parse_args(sys.argv[2:])
        result = experiment.run(args.session, args.trial, args.base_frame, args.odom_topic,
                                transfer=json.loads(args.packet.read_text()))
        print(json.dumps(result, indent=2), flush=True)
        raise SystemExit(0 if result.get('success') is True else 1)
    parser.add_argument('--map', type=Path, required=True, help='Must match the map served by localization')
    parser.add_argument('--conditions', nargs='+', choices=['E1', *experiment.CONDITIONS])
    parser.add_argument('--repeats', type=int, nargs='+')
    parser.add_argument('--resume', action='store_true',
                        help='Skip only the successful recorded prefix; align to the next frozen start before continuing')
    parser.add_argument('--continue-on-endpoint-failure', action='store_true',
                        help='Record verified stopped endpoint misses as failures and continue; also permits resuming past them')
    parser.add_argument('--start-from-current', action='store_true',
                        help='Normal E1 batches align then reanchor; retries turn back between current-pose trials')
    parser.add_argument('--dry-run', action='store_true', help='Validate frozen files and static-map geometry only; no ROS nodes/goals')
    args = parser.parse_args()
    with (args.session / '.batch.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another batch owns this session')
        manifest, trials = selected_trials(args.session, args.conditions, args.repeats, args.resume,
                                          args.continue_on_endpoint_failure)
        if 'retry_of' in manifest:
            args.start_from_current = True
        if not trials:
            print('All selected trials already recorded. No motion requested.')
            return
        if args.start_from_current:
            if not manifest.get('bidirectional') or any(t['task']=='E2_environment' for t in trials):
                raise ValueError('Per-trial current starts require a bidirectional E1 schedule')
            data = load_map(args.map)
        else:
            data = check_geometry(args.session, trials, args.map)
        print(f'{len(trials)} retries from current stopped poses; turn back along the preceding actual path between trials.'
              if 'retry_of' in manifest else
              f'{len(trials)} scored legs, alternating forward/reverse; only start alignment is unscored.'
              if manifest.get('bidirectional') else
              f'{len(trials)} trials; each followed by an unscored return. Last return goes to the first start.')
        for trial in trials:
            direction = trial.get('direction','forward')
            print(trial['id'], f'(source direction: {direction}; current-pose retry)' if 'retry_of' in manifest else direction)
        if args.resume and 'retry_of' in manifest:
            print(f'First pending retry: use the current heading if more than {RETRY_RESUME_RELOCATION_DISTANCE_M:.1f} m '
                  'from the previous stopped position; otherwise turn back. Live pose decides before motion.')
        if args.dry_run:
            print('Schedule and map loaded. Live path placements will be checked before each trial; no goals sent.'
                  if args.start_from_current else
                  'Static plan checked. No live localization, sensor, or motion verification performed.')
            return
        output = args.session.resolve() / 'batches' / (time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6])
        output.mkdir(parents=True)
        experiment.write_json(output / 'map_input.json', dict(path=str(args.map.resolve()),
            yaml_sha256=experiment.digest(args.map),
            image_sha256=experiment.digest(args.map.parent / data[0]['image'])))
        execute(args, manifest, trials, data, output)
        print(f'Completed schedule: {output}')


if __name__ == '__main__':
    main()
