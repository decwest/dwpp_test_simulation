#!/usr/bin/env python3
"""One-terminal mapping, current-pose E1 and prepared map-fixed E2 experiments."""
import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import re
import select
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid
from zoneinfo import ZoneInfo

import yaml

import dwvp_access_experiment as experiment
from dwvp_access_path import load_map


def timestamp():
    return datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y%m%d_%H%M%S')


def new_directory(parent, prefix, name=None):
    parent.mkdir(parents=True, exist_ok=True)
    stem = name or f'{prefix}_{timestamp()}'
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', stem):
        raise ValueError('Use letters, digits, underscores, dots or hyphens for the environment name')
    for suffix in range(10000):
        path = parent/(stem if suffix == 0 else f'{stem}_{suffix:02d}')
        try:
            path.mkdir()
            return path
        except FileExistsError:
            if name is not None:
                raise FileExistsError(f'Environment already exists: {path}')
    raise RuntimeError('Could not reserve a fresh timestamped directory')


def atomic_json(path, data):
    temp = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        temp.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def checked_map(path):
    path = Path(path).resolve()
    info, pixels, blocked = load_map(path)
    if pixels.size == 0 or blocked.all():
        raise ValueError(f'Map has no free cells: {path}')
    # A trinary saver uses gray 205 for unknown; the old .25 threshold made it free.
    if (not 0 < float(info['free_thresh']) < float(info['occupied_thresh']) < 1
            or float(info['resolution']) <= 0):
        raise ValueError(f'Invalid map thresholds/resolution: {path}')
    if not info.get('negate', 0) and float(info['free_thresh']) > (255-205)/255:
        raise ValueError(f'Unknown gray 205 would become free: {path}; use free_thresh: 0.196')
    return info


def choose_map(workspace, requested=None):
    if requested is None:
        pointer = workspace/'maps/latest.json'
        if not pointer.exists():
            raise ValueError('No saved mapping session yet. Run mapping, or pass experiment --map PATH.')
        requested = workspace/'maps'/json.loads(pointer.read_text())['map']
    else:
        requested = Path(requested)
        if not requested.is_absolute():
            requested = workspace/requested
    checked_map(requested)
    return requested.resolve()


def copy_map(source, folder):
    info = checked_map(source)
    folder.mkdir()
    image = source.parent/info['image']
    destination = folder/('map'+image.suffix)
    shutil.copy2(image, destination)
    info['image'] = destination.name
    saved = folder/'map.yaml'
    saved.write_text(yaml.safe_dump(info, sort_keys=False))
    return saved


def fixed_environment(manifest):
    return {t['task'] for t in manifest['trials']} == {'E2_environment'}


def verify_environment_map(session, saved_map, manifest):
    """Bind the offline route to the exact saved map used for localization."""
    metadata = json.loads((session/'environment_path_metadata.json').read_text())
    info = checked_map(saved_map)
    if (metadata['csv_sha256'] != manifest['paths']['E2_environment']['sha256']
            or metadata['map_yaml_sha256'] != experiment.digest(saved_map)
            or metadata['map_image_sha256'] != experiment.digest(saved_map.parent/info['image'])):
        raise ValueError('Prepared E2 route/map differs from the offline planner metadata')


def prepared_inputs(workspace, requested):
    """Validate an unrecorded E2 session and its map before starting any nodes."""
    from dwvp_access_batch import check_geometry, selected_trials
    session = Path(requested)
    session = (session if session.is_absolute() else workspace/session).resolve()
    manifest, trials = selected_trials(session)
    if not fixed_environment(manifest) or 'retry_of' in manifest:
        raise ValueError('--prepared-session requires a map-fixed E2 session')
    saved_map = session.parent/'map/map.yaml'
    verify_environment_map(session, saved_map, manifest)
    check_geometry(session, trials, saved_map)
    return session, saved_map, trials


def fixed_retry_inputs(workspace, requested, trial_id):
    """Select an explicitly requested E2 attempt, including an interrupted one."""
    from dwvp_access_batch import check_geometry
    session = Path(requested)
    session = (session if session.is_absolute() else workspace/session).resolve()
    manifest, trial, _ = experiment.load_trial(session, trial_id)
    if not fixed_environment(manifest) or manifest.get('bidirectional'):
        raise ValueError('Fixed E2 reacquisition requires a one-way E2-only session')
    folder = session/'runs'/trial_id
    result = json.loads((folder/'result.json').read_text())
    recorded = json.loads((folder/'trial.json').read_text())
    if (recorded.get('trial') != trial
            or recorded.get('manifest_sha256') != experiment.digest(session/'manifest.json')
            or recorded.get('params_sha256') != trial['params_sha256']):
        raise ValueError('Source attempt differs from its frozen E2 inputs')
    if result.get('status') not in ('succeeded', 'failed', 'interrupted', 'setup_failed'):
        raise ValueError('Source E2 attempt has no terminal result')
    saved_map = session.parent/'map/map.yaml'
    verify_environment_map(session, saved_map, manifest)
    check_geometry(session, [trial], saved_map)
    return session, saved_map, [trial]


def prepare_fixed_retry(source, output, trial_id, *, template=None):
    """New fixed-path attempt with current HSR settings and explicit provenance.

    The old attempt is never edited or treated as a successful resume prefix.
    Changed settings remain distinguishable when analyzing experimental data.
    """
    source, output = Path(source), Path(output)
    manifest, trial, _ = experiment.load_trial(source, trial_id)
    config = yaml.safe_load((source/'experiment_config.yaml').read_text())
    config['conditions']['E2_environment'].update(methods=[trial['controller']],
        start_pose=experiment.start_pose(manifest, trial).tolist())
    output.parent.mkdir(parents=True, exist_ok=True)
    settings = output.parent/'retry_config.yaml'
    if settings.exists():
        raise FileExistsError(settings)
    settings.write_text(yaml.safe_dump(config, sort_keys=False))
    if template is None:
        template = Path(__file__).resolve().parents[1]/'params/hsrb_dwvp_access_params.yaml'
        if not template.is_file():
            from ament_index_python.packages import get_package_share_directory
            template = Path(get_package_share_directory('dwpp_test_simulation'))/'params/hsrb_dwvp_access_params.yaml'
    new = experiment.prepare(output, template, origin=manifest['map_origin'],
        environment_path=source/manifest['paths']['E2_environment']['file'],
        seed=manifest['seed'], config_file=settings, conditions=['E2_environment'], repeats=[trial['repeat']])
    if new['paths']['E2_environment']['sha256'] != manifest['paths']['E2_environment']['sha256']:
        raise ValueError('Fixed retry changed the source reference CSV')
    shutil.copy2(source/'environment_path_metadata.json', output/'environment_path_metadata.json')
    for name, path in [('source_manifest.json', source/'manifest.json'),
                       ('source_result.json', source/'runs'/trial_id/'result.json'),
                       ('source_params.yaml', source/trial['params_file'])]:
        shutil.copy2(path, output.parent/name)
    before = yaml.safe_load((source/trial['params_file']).read_text())
    after = yaml.safe_load((output/new['trials'][0]['params_file']).read_text())
    experiment.write_json(output.parent/'reacquisition.json', dict(source_session=str(source),
        trial_id=trial_id, source_manifest_sha256=experiment.digest(source/'manifest.json'),
        source_result_sha256=experiment.digest(source/'runs'/trial_id/'result.json'),
        old_params_sha256=trial['params_sha256'], new_params_sha256=new['trials'][0]['params_sha256'],
        changed_parameter_sections=[k for k in sorted(set(before)|set(after)) if before.get(k)!=after.get(k)],
        reference_policy='same_frozen_map_path', source_attempt_preserved=True,
        settings_policy='current HSR template; do not silently pool changed settings with source data'))
    return new


def copy_fixed_map(source, folder):
    """Preserve the exact YAML bytes covered by E2 planner metadata."""
    info = checked_map(source)
    image = Path(info['image'])
    if image.is_absolute() or '..' in image.parts:
        raise ValueError('Fixed E2 map image must be packaged beside its YAML')
    folder.mkdir(parents=True, exist_ok=False)
    (folder/image).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source.parent/image, folder/image)
    shutil.copy2(source, folder/'map.yaml')
    return folder/'map.yaml'


def resume_inputs(workspace, requested):
    """Read and validate a one-terminal run without changing any recorded files."""
    from dwvp_access_batch import selected_trials
    if str(requested) == 'latest':
        requested = Path(json.loads((workspace/'results/dwvp_access/latest.json').read_text())['session'])
    session = Path(requested)
    session = (session if session.is_absolute() else workspace/session).resolve()
    manifest, pending = selected_trials(session, resume=True, continue_on_endpoint_failure=True)
    is_e2 = fixed_environment(manifest)
    if not is_e2 and (not manifest.get('bidirectional') or any(t['task'] == 'E2_environment' for t in manifest['trials'])):
        raise ValueError('One-terminal resume requires a bidirectional E1 or map-fixed E2-only session')
    saved_map = session.parent/'map/map.yaml'
    info = checked_map(saved_map)
    map_records = list((session/'batches').glob('*/map_input.json'))
    if is_e2:
        from dwvp_access_batch import check_geometry
        verify_environment_map(session, saved_map, manifest)
        if pending:
            check_geometry(session, pending, saved_map)
    if not map_records and not is_e2:
        raise ValueError('Cannot verify the previous batch map; use the explicit batch workflow')
    for path in map_records:
        recorded = json.loads(path.read_text())
        if (recorded['yaml_sha256'] != experiment.digest(saved_map)
                or recorded['image_sha256'] != experiment.digest(saved_map.parent/info['image'])):
            raise ValueError('Saved map changed since the previous batch')
    return session, saved_map, pending


class Process:
    """A process group owned by this workflow; children never read its terminal."""
    def __init__(self, command, logfile):
        self.logfile = logfile
        self.stopped = False
        self.stream = logfile.open('w')
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=self.stream,
                stderr=subprocess.STDOUT, start_new_session=True, env=dict(os.environ, PYTHONUNBUFFERED='1'))
        except BaseException:
            self.stream.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()

    def signal(self, sig):
        try:
            os.killpg(self.process.pid, sig)
        except ProcessLookupError:
            pass

    def stop(self):
        if self.stopped:
            return
        if self.process.poll() is None:
            self.signal(signal.SIGINT)
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.signal(signal.SIGTERM)
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.signal(signal.SIGKILL)
                    self.process.wait(timeout=5)
        self.signal(signal.SIGTERM)  # Also clean up any surviving group children.
        self.stream.close()
        self.stopped = True

    def check(self):
        if self.process.poll() is not None:
            tail = '\n'.join(self.logfile.read_text().splitlines()[-15:])
            raise RuntimeError(f'Process exited ({self.process.returncode}); see {self.logfile}\n{tail}')


def check_background(background):
    for process in background:
        process.check()


def wait_enter(message, background):
    print(message, flush=True)
    while True:
        check_background(background)
        if select.select([sys.stdin], [], [], .2)[0]:
            if not sys.stdin.readline():
                raise RuntimeError('Terminal input closed; workflow canceled')
            check_background(background)
            return


def run_child(command, logfile, background=(), timeout=None):
    check_background(background)
    print('$ '+shlex.join(map(str, command)), flush=True)
    with Process(command, logfile) as child, logfile.open() as reader:
        begin = time.monotonic()
        while True:
            check_background(background)
            output = reader.read()
            if output:
                print(output, end='', flush=True)
            code = child.process.poll()
            if code is not None:
                print(reader.read(), end='', flush=True)
                if code != 0:
                    raise RuntimeError(f'Command failed ({code}); see {logfile}')
                return
            if timeout is not None and time.monotonic()-begin > timeout:
                raise TimeoutError(f'Command timed out; see {logfile}')
            time.sleep(.1)


def inspect_idle(node, allow_robot_joy=False):
    """Return whether standard JOY may need handoff; reject other conflicts first."""
    conflicts = {'slam_toolbox', 'amcl', 'map_server', 'controller_server', 'velocity_smoother', 'planner_server'}
    active = conflicts.intersection(name for name, _ in node.get_node_names_and_namespaces())
    if active:
        raise RuntimeError(f'Stop existing nodes first: {sorted(active)}')
    pending = False
    for topic in ('/map', '/cmd_vel_nav', '/omni_base_controller/cmd_vel', '/joy'):
        publishers = node.get_publishers_info_by_topic(topic)
        if not publishers:
            continue
        allowed = {'/omni_base_controller/cmd_vel': {'joystick_control_node'},
                   '/joy': {'joy_node', 'joy_linux_node'}}.get(topic, set())
        if allow_robot_joy and all(p.node_namespace in ('', '/') and p.node_name in allowed for p in publishers):
            pending = True
            continue
        counts = Counter(p.node_namespace.rstrip('/')+'/'+p.node_name for p in publishers)
        details = ', '.join(f'{name}: {count} publisher(s)' for name, count in sorted(counts.items()))
        raise RuntimeError(f'Stop existing mapping/navigation/teleop first: {topic} [{details}]. '
                           'PC-side or nonstandard teleop must be stopped in its own terminal. '
                           'See docs/dwvp_access_hsr.md.')
    return pending


def stopped_wheels(node, timeout=8.):
    """Check fresh arrivals and progressing odometry before stopping teleop.

    This observation only authorizes stopping existing input processes. Map pose
    capture and scored motion retain their stricter source-clock freshness checks.
    """
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.qos import qos_profile_sensor_data
    latest = []
    def receive(msg):
        stamp = msg.header.stamp.sec+msg.header.stamp.nanosec/1e9
        progresses = bool(latest and stamp>latest[2])
        latest[:] = [msg, time.monotonic(), stamp, progresses]
    subscription = node.create_subscription(Odometry, '/omni_base_controller/wheel_odom', receive, qos_profile_sensor_data)
    end, stable = time.monotonic()+timeout, None
    try:
        while time.monotonic()<end:
            rclpy.spin_once(node, timeout_sec=.02)
            if not latest:
                continue
            msg, received, _, progresses = latest
            age = node.get_clock().now().nanoseconds/1e9 - (msg.header.stamp.sec+msg.header.stamp.nanosec/1e9)
            velocity = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.angular.z]
            stationary = (all(math.isfinite(v) for v in velocity) and math.hypot(*velocity[:2])<=.005
                          and abs(velocity[2])<=.01 and time.monotonic()-received<=.2
                          and progresses)
            if not stationary:
                stable = None
            elif stable is None:
                stable = time.monotonic()
            elif time.monotonic()-stable>=.5:
                return dict(velocity=velocity, odom_source_age_s=age, stationary_interval_s=.5,
                            odom_receive_age_s=time.monotonic()-received,
                            source_clock_synchronized=bool(experiment.source_age_is_fresh(age)))
        raise RuntimeError('Release LB and stop the robot before JOY handoff; fresh stationary wheel odometry is required')
    finally:
        node.destroy_subscription(subscription)


def require_idle(robot=None):
    """Optionally hand off standard robot JOY, then require exclusive PC control."""
    import rclpy
    from rclpy.node import Node
    rclpy.init()
    node = Node('dwvp_workflow_check')
    try:
        end = time.monotonic()+2.
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=.05)
        auto_stop = robot is not None and not robot.no_stop_robot_joy
        if not inspect_idle(node, allow_robot_joy=auto_stop):
            return None
        stationary = stopped_wheels(node)
        if not stationary['source_clock_synchronized']:
            print(f'本体とPCの時刻差を確認してください（odom age {stationary["odom_source_age_s"]:.3f} s）。'
                  'JOY停止は行いますが、地図保存・実験の前に時刻同期が必要です。', flush=True)
        from dwvp_access_robot_teleop import stop_robot_teleop
        print(f'本体 {robot.robot_host} の標準JOYを停止し、PC側へ切り替えます。', flush=True)
        audit = robot.workspace/'log/robot_teleop'
        audit.mkdir(parents=True, exist_ok=True)
        report = dict(host=robot.robot_host, container=robot.robot_container, stationary=stationary)
        try:
            report.update(stop_robot_teleop(robot.robot_host, robot.robot_user, robot.robot_container))
            end = time.monotonic()+5.
            while True:
                rclpy.spin_once(node, timeout_sec=.1)
                try:
                    inspect_idle(node)
                    break
                except RuntimeError:
                    if time.monotonic()>=end:
                        raise
            report['status'] = 'stopped_and_graph_clear'
            print(f'本体JOY停止: {len(report["stopped"])} プロセス。既存の速度指令・JOY送信元がないことを確認しました。', flush=True)
            return report
        except Exception as exc:
            report.update(status='failed', error=str(exc))
            raise
        finally:
            atomic_json(audit/(timestamp()+'_'+uuid.uuid4().hex[:6]+'.json'), report)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def stopped_pose():
    import rclpy
    from dwvp_access_runtime import Observer
    rclpy.init()
    observer = Observer()
    try:
        capture = observer.stopped()
        end = time.monotonic()+10.
        while True:
            try:
                observer.command_owners(False)
                return capture
            except RuntimeError:
                if time.monotonic() >= end:
                    raise
                rclpy.spin_once(observer, timeout_sec=.05)
                capture = observer.stopped()
    finally:
        observer.destroy_node()
        rclpy.try_shutdown()


def joy_command():
    return ['ros2', 'launch', 'ytlab2_hsr_modules', 'joy_teleop.launch.py', 'shutdown_on_exit:=true']


def mapping(args, folder):
    command = ['ros2', 'launch', 'ytlab2_hsr_modules', 'mapping_joy.launch.py',
               'use_sim_time:=false', 'use_joy:=false', f'use_rviz:={str(not args.no_rviz).lower()}']
    with ExitStack() as owned:
        slam = owned.enter_context(Process(command, folder/'mapping.log'))
        joy = None if args.no_joy else owned.enter_context(Process(joy_command(), folder/'joy.log'))
        background = [p for p in (slam, joy) if p is not None]
        print(f'地図の保存先: {folder}\n起動ログ: {folder}/mapping.log', flush=True)
        wait_enter('F310 の LB を押して地図を作成してください。完成したら LB を離して停止し、Enter で保存・終了します。', background)
        if joy:
            joy.stop()
        attempt = 0
        while True:
            check_background([slam])
            attempt += 1
            try:
                capture = stopped_pose()
                run_child(['ros2', 'run', 'nav2_map_server', 'map_saver_cli', '-f', str(folder/'map'),
                           '--fmt', 'pgm', '--mode', 'trinary', '--free', '0.196', '--occ', '0.65',
                           '--ros-args', '-p', 'save_map_timeout:=10.0'],
                          folder/f'save_{attempt:02d}.log', [slam], timeout=30)
                info = checked_map(folder/'map.yaml')
                break
            except (RuntimeError, TimeoutError, ValueError, OSError) as exc:
                check_background([slam])
                print(f'地図を保存できませんでした: {exc}', flush=True)
                wait_enter('SLAM は継続しています。原因を解消して Enter で保存を再試行、Ctrl-C で中止します。', [slam])
        experiment.write_json(folder/'environment.json', dict(name=folder.name, saved_at=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
            map_sha256=experiment.digest(folder/'map.yaml'), image_sha256=experiment.digest(folder/info['image']),
            stopped_pose=capture))
        atomic_json(folder.parent/'latest.json', dict(environment=folder.name, map=f'{folder.name}/map.yaml'))
    print(f'保存しました: {folder}/map.yaml\n次は ./dwvp_access.sh experiment で、この地図の新規実験を開始できます。', flush=True)


def is_retry(args):
    return getattr(args, 'retry_failed', None) is not None or getattr(args, 'retry_trial', None) is not None


def run_experiment(args, folder, source):
    resuming = getattr(args, 'resume', None) is not None
    prepared = getattr(args, 'prepared_session', None) is not None
    retrying = is_retry(args)
    session = args.resume_session if resuming else args.prepared_session if prepared else folder/'session'
    is_e2 = (resuming or prepared) and fixed_environment(json.loads((session/'manifest.json').read_text()))
    if retrying:
        is_e2 = bool(getattr(args, 'fixed_retry', False))
        if is_e2:
            retry = prepare_fixed_retry(args.retry_source, session, args.retry_trial)
        else:
            from dwvp_access_batch import prepare_retry
            retry = prepare_retry(args.retry_source, session, trial_id=getattr(args, 'retry_trial', None))
        args.params = session/retry['trials'][0]['params_file']
    current_pose_retry = (retrying and not is_e2) or (resuming and 'retry_of' in json.loads((session/'manifest.json').read_text()))
    saved_map = source if resuming or prepared else (copy_fixed_map if is_e2 else copy_map)(source, folder/'map')
    experiment.write_json(folder/'map_input.json', dict(source=str(source), map=str(saved_map),
        yaml_sha256=experiment.digest(source), image_sha256=experiment.digest(source.parent/checked_map(source)['image'])))
    localize = ['ros2', 'launch', 'dwpp_test_simulation', 'dwvp_access_hsr.launch.py',
        f'params_file:={args.params}', f'map:={saved_map}', 'localization_only:=true',
        'use_sim_time:=false', 'shutdown_on_exit:=true', f'use_rviz:={str(not args.no_rviz).lower()}']
    with ExitStack() as owned:
        localization = owned.enter_context(Process(localize, folder/'localization.log'))
        preview = None
        if is_e2:
            from dwvp_access_batch import selected_trials
            manifest, pending = selected_trials(session, resume=True, continue_on_endpoint_failure=True)
            next_trial = pending[0]
            preview = owned.enter_context(Process(
                ['ros2', 'run', 'dwpp_test_simulation', 'dwvp_access_experiment.py', 'preview',
                 '--session', str(session), '--trial', next_trial['id']], folder/'preview.log'))
        joy = None if args.no_joy else owned.enter_context(Process(joy_command(), folder/'joy.log'))
        background = [p for p in (localization, preview, joy) if p is not None]
        label = '再試行' if retrying else ('再開' if resuming else '新規実験')
        print(f'地図: {source}\n{label}: {session}\n起動ログ: {folder}/localization.log', flush=True)
        if is_e2:
            start = experiment.start_pose(manifest, next_trial)
            pattern = '固定経路の往復試行' if manifest.get('bidirectional') else '同じS→G経路の試行（各試行後にSへ帰還、帰還は計測外）'
            print(f'E2 は地図上に固定した経路です。残り {len(pending)} 試行、次は {next_trial["id"]} ({next_trial.get("direction", "forward")})。\n'
                  f'開始姿勢: x={start[0]:.3f} m, y={start[1]:.3f} m, yaw={start[2]:.3f} rad。\n'
                  'RViz に経路と開始姿勢を表示します。F310 で開始位置付近へ移動してください。\n'
                  f'Enter 後に開始姿勢へ位置合わせして、{pattern}を開始します。', flush=True)
        if current_pose_retry:
            print('再試行は現在位置基準です。2件目以降は直前の経路を戻る方向へその場で向き直り、停止後に経路を作ります。\n'
                  '再開の最初の1件は、前回の停止位置から0.5 m以内なら折返し、離れていれば現在の位置・向きから始めます。\n'
                  'F310で経路が収まる位置・向きに合わせ、LBを離してください。', flush=True)
        wait_enter('RViz の 2D Pose Estimate で初期位置を設定し、レーザと地図を合わせてください。\n'
                   'LB を離して停止してください。Enter で選択した試行を開始します。' if resuming or retrying or prepared else
                   'RViz の 2D Pose Estimate で初期位置を設定し、レーザと地図を合わせてください。\n'
                   'F310 で開始場所へ移動し、LB を離して停止してください。Enter で新規の往復実験を開始します。', background)
        if joy:
            joy.stop()
        if preview:
            preview.stop()
        capture = stopped_pose()
        check_background([localization])
        if not resuming and not retrying and not prepared:
            prepare = ['ros2', 'run', 'dwpp_test_simulation', 'dwvp_access_experiment.py', 'prepare',
                       '--output', str(session), '--params', str(args.params),
                       '--start-from-current', '--bidirectional', '--conditions', *args.conditions]
            if args.config:
                prepare += ['--config', str(args.config)]
            if args.repeats:
                prepare += ['--repeats', *map(str, args.repeats)]
            run_child(prepare, folder/'prepare.log', [localization], timeout=60)
        experiment.write_json(folder/'ready_pose.json', capture)
        atomic_json(args.workspace/'results/dwvp_access/latest.json',
                    dict(session=str(session.relative_to(args.workspace)), map=str(saved_map.relative_to(args.workspace))))
        batch = ['ros2', 'run', 'dwpp_test_simulation', 'dwvp_access_batch.py',
                 '--session', str(session), '--map', str(saved_map),
                 '--continue-on-endpoint-failure']
        if not is_e2:
            batch += ['--start-from-current']
        if resuming or prepared or is_e2:
            # --resume also aligns an entirely unrecorded fixed-path session to
            # its first start, without changing its reference or trial order.
            batch += ['--resume']
        run_child(batch+['--dry-run'], folder/'plan.log', [localization], timeout=60)
        run_child(batch, folder/'batch.log', [localization])
        # The batch already summarizes on completion and failure. Show its final status.
        report = experiment.summarize(session)
        succeeded = sum(t['success'] for t in report['trials'])
        print(f"完了: 記録 {report['recorded']}/{report['planned']}、成功 {succeeded}、"
              f"失敗 {report['recorded']-succeeded}、未実施 {report['pending']}\n結果: {session}", flush=True)
        if retrying:
            print(f'再試行のみの集計です。元の実験結果は保持しています: {args.retry_source}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('mapping', help='SLAM/RViz/F310; Enter saves and closes everything')
    p.add_argument('--name', help='Optional environment name; existing names are never overwritten')
    e = sub.add_parser('experiment', help='Localize and run fresh E1 or prepared map-fixed E2 trials')
    previous = e.add_mutually_exclusive_group()
    previous.add_argument('--prepared-session', type=Path,
                   help='Start a prepared map-fixed E2 session; uses its adjacent map/map.yaml and frozen route')
    previous.add_argument('--resume', nargs='?', const='latest', type=Path,
                   help='Resume the latest or specified session; preserve all recorded successes and stopped endpoint misses')
    previous.add_argument('--retry-failed', nargs='?', const='latest', type=Path,
                   help='Repeat failed conditions from each current pose in a separate attempt, without returning to old starts')
    previous.add_argument('--retry-trial', metavar='TRIAL_ID',
                   help='Reacquire one trial; E2 also accepts interrupted attempts and uses current HSR settings on the frozen path')
    e.add_argument('--source-session', type=Path,
                   help='Original session for --retry-trial; its results and frozen conditions are preserved')
    e.add_argument('--map', type=Path, help='Override maps/latest.json; relative to the workspace')
    e.add_argument('--params', type=Path, help='Controller template; default is the installed HSR template')
    e.add_argument('--config', type=Path, help='Optional experimental condition configuration')
    e.add_argument('--conditions', nargs='+', choices=['E1', *experiment.CONDITIONS[:-1]],
                   help='Default: all E1 conditions; E2 requires the separate map-fixed workflow')
    e.add_argument('--repeats', nargs='+', type=int, choices=range(1, 6), help='Default: all five repeats, 55 E1 legs')
    for command in (p, e):
        command.add_argument('--no-rviz', action='store_true', help='Do not start RViz (e.g. headless verification)')
        command.add_argument('--no-joy', action='store_true', help='Do not start the F310 driver')
        command.add_argument('--no-stop-robot-joy', action='store_true',
                             help='Disable automatic robot teleop shutdown; existing publishers still block startup')
        command.add_argument('--robot-host', default=os.environ.get('HSR_IP', '192.168.50.10'))
        command.add_argument('--robot-user', default=os.environ.get('HSR_SSH_USER', 'administrator'))
        command.add_argument('--robot-container', default=os.environ.get('HSR_ROBOT_CONTAINER', 'docker.humble.robot.service'))
        command.add_argument('--dry-run', action='store_true', help='Show inputs without starting nodes or creating a session')
    args = parser.parse_args()
    args.workspace = args.workspace.resolve()
    source = None
    if args.command == 'experiment':
        if (args.retry_trial is not None) != (args.source_session is not None):
            parser.error('--retry-trial and --source-session must be supplied together')
        if args.prepared_session is not None:
            if any(getattr(args, name) is not None for name in ('map', 'params', 'config', 'conditions', 'repeats')):
                raise ValueError('--prepared-session uses the saved map and frozen conditions; do not supply input overrides')
            selected, source, pending = prepared_inputs(args.workspace, args.prepared_session)
            args.prepared_session = selected
            args.params = selected/pending[0]['params_file']
            manifest = json.loads((selected/'manifest.json').read_text())
            pattern = ('alternating forward/reverse' if manifest.get('bidirectional') else
                       'all trials S -> G; unscored return to S after each trial, including the last')
            print(f'Prepared E2: {selected}\nTrials: {len(pending)}; fixed map path; {pattern}.')
        elif args.resume is not None or is_retry(args):
            if any(getattr(args, name) is not None for name in ('map', 'params', 'config', 'conditions', 'repeats')):
                raise ValueError('--resume / --retry-failed / --retry-trial use the saved map and frozen conditions; do not supply input overrides')
            args.fixed_retry = False
            if args.retry_trial is not None:
                candidate = args.source_session
                candidate = (candidate if candidate.is_absolute() else args.workspace/candidate).resolve()
                args.fixed_retry = fixed_environment(json.loads((candidate/'manifest.json').read_text()))
            selected, source, pending = (fixed_retry_inputs(args.workspace, args.source_session, args.retry_trial)
                if args.fixed_retry else resume_inputs(args.workspace, args.resume or args.retry_failed or args.source_session))
            if is_retry(args):
                if not args.fixed_retry and fixed_environment(json.loads((selected/'manifest.json').read_text())):
                    raise ValueError('Current-pose retries are E1-only; E2 reacquisition must retain the fixed map path')
                if not args.fixed_retry:
                    from dwvp_access_batch import failed_trials, explicit_retry_trial
                    _, pending = (explicit_retry_trial(selected, args.retry_trial) if args.retry_trial is not None
                                  else failed_trials(selected))
                args.retry_source = selected
                label = 'Explicit trials to retry' if args.retry_trial is not None else 'Failed trials to retry'
                print(f'Retry source: {selected}\n{label}: {len(pending)}; original results are preserved.')
                for trial in pending:
                    if args.fixed_retry:
                        print(f"  {trial['id']} (same frozen S -> G; new HSR settings recorded separately)")
                    else:
                        print(f"  {trial['id']} (source direction: {trial['direction']}; retry uses current pose)")
            else:
                args.resume_session = selected
                print(f'Resume: {selected}\nRemaining trials: {len(pending)}; existing attempts are preserved.')
            if not pending:
                print('No trials to execute. No ROS nodes or motion requested.')
                return
            args.params = selected/pending[0]['params_file']
        else:
            from ament_index_python.packages import get_package_share_directory
            args.params = args.params or Path(get_package_share_directory('dwpp_test_simulation'))/'params/hsrb_dwvp_access_params.yaml'
            args.conditions = args.conditions or ['E1']
        for name in ('params', 'config'):
            value = getattr(args, name)
            if value is not None:
                value = value if value.is_absolute() else args.workspace/value
                if not value.is_file():
                    raise FileNotFoundError(value)
                setattr(args, name, value.resolve())
        if args.resume is None and not is_retry(args) and args.prepared_session is None:
            source = choose_map(args.workspace, args.map)
    if args.dry_run:
        if getattr(args, 'prepared_session', None) is not None:
            print(f'Map: {source}\nNext trial: {pending[0]["id"]} ({pending[0].get("direction", "forward")})\n'
                  'Fixed E2 route checked against map; align to the first start after Enter; no current-pose reanchoring.')
        elif is_retry(args):
            print(f'Map: {source}\nNew attempt: {args.retry_source.parent}/retries/retry_<JST timestamp>/session')
            print('Same frozen map path; current HSR parameters; return to S after the trial; source data preserved.'
                  if getattr(args, 'fixed_retry', False) else
                  'Conditions preserved; current-pose retries turn back along the preceding actual path between trials.')
        elif getattr(args, 'resume', None) is not None:
            print(f'Map: {source}\nNext trial: {pending[0]["id"]} ({pending[0].get("direction", "forward")})')
            if 'retry_of' in json.loads((args.resume_session/'manifest.json').read_text()):
                print('Retry start policy: turn back between trials. On resume, the first pending leg uses the '
                      'current heading if repositioned more than 0.5 m from the previous stopped position.')
            print('Stopped endpoint misses remain failed; timeout, abort, sensor and stop failures block continuation.')
        elif source:
            print(f'Map: {source}\nNew session: results/dwvp_access/{source.parent.name}/run_<JST timestamp>/session\n'
                  f'Conditions={args.conditions}; repeats={args.repeats or [1,2,3,4,5]}; bidirectional; align then reanchor at each measured stop.')
        else:
            print(f'Map output: maps/{args.name or "lab_<JST timestamp>"}/map.yaml\nSLAM Toolbox + mapping RViz + F310; Enter saves.')
        print('No ROS nodes, motion goals, SSH connections, process stops, or new session files created.')
        return
    if not sys.stdin.isatty():
        raise RuntimeError('Run in an interactive terminal (docker exec -it); Enter is required after mapping/initial localization.')
    args.workspace.mkdir(parents=True, exist_ok=True)
    with (args.workspace/'.dwvp_workflow.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another mapping/experiment workflow owns this workspace')
        handoff = require_idle(args)
        resuming = getattr(args, 'resume', None) is not None
        prepared = getattr(args, 'prepared_session', None) is not None
        retrying = is_retry(args)
        parent = (args.retry_source.parent/'retries' if retrying else
                  args.resume_session.parent/'resumes' if resuming else
                  args.prepared_session.parent/'executions' if prepared else
                  args.workspace/('maps' if source is None else f'results/dwvp_access/{source.parent.name}'))
        prefix = 'retry' if retrying else ('resume' if resuming else ('start' if prepared else ('lab' if source is None else 'run')))
        folder = new_directory(parent, prefix, getattr(args, 'name', None))
        state = dict(command=args.command, status='starting', folder=str(folder), robot_teleop_handoff=handoff)
        if resuming:
            state['resume_session'] = str(args.resume_session)
        if prepared:
            state['prepared_session'] = str(args.prepared_session)
        if retrying:
            state['retry_source'] = str(args.retry_source)
            if args.retry_trial is not None:
                state['retry_trial'] = args.retry_trial
        experiment.write_json(folder/'workflow.json', state)
        try:
            if args.command == 'mapping':
                mapping(args, folder)
            else:
                run_experiment(args, folder, source)
            state['status'] = 'completed'
        except BaseException as exc:
            state.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc))
            print(f'停止しました。記録: {folder}', flush=True)
            raise
        finally:
            experiment.write_json(folder/'workflow.json', state)


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Workflow interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
