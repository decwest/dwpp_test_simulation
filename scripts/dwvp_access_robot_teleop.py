"""Stop only standard boot_app joystick children in the selected HSR container.

The same stdlib-only source is sent over SSH to `docker exec ... python3 -`.
No file or persistent boot setting is changed on the robot.
"""
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import time


JOY_DRIVER = '/opt/ros/humble/lib/joy_linux/joy_linux_node'
JOY_CONTROL = '/root/ros2_ws/install/hsrb_joystick_teleop/lib/hsrb_joystick_teleop/joystick_control_node'


def read_process(proc, pid):
    folder = Path(proc)/str(pid)
    args = folder.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')
    fields = folder.joinpath('stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=int(pid), args=args, parent=int(fields[1]), started=fields[19],
                cgroup=folder.joinpath('cgroup').read_text())


def standard_teleop(proc='/proc'):
    proc = Path(proc)
    own_group = (proc/'self/cgroup').read_text()
    found = []
    for folder in proc.iterdir():
        if not folder.name.isdigit():
            continue
        try:
            item = read_process(proc, folder.name)
            args = item['args']
            if args[0] == JOY_DRIVER:
                role = 'joy_driver'
            elif (args[0] == JOY_CONTROL or
                  (args[0] in ('/usr/bin/python3', '/usr/bin/python3.10') and args[1:2] == [JOY_CONTROL])):
                role = 'joystick_control'
            else:
                continue
            if item['cgroup'] != own_group:
                continue
            parent = read_process(proc, item['parent'])
            if parent['cgroup'] != own_group:
                continue
            expected = ['launch', 'hsrb_robot_launch', 'boot_app.launch.py']
            if not any(parent['args'][i:i+3] == expected for i in range(len(parent['args']))):
                continue
            item['role'] = role
            found.append(item)
        except (FileNotFoundError, ProcessLookupError):
            continue  # A process exited during the scan.
    return sorted(found, key=lambda item: (item['role'] != 'joy_driver', item['pid']))


def interrupt_verified(item, proc='/proc'):
    """A pidfd prevents signaling an unrelated process if a PID is recycled."""
    try:
        fd = os.pidfd_open(item['pid'])
    except ProcessLookupError:
        return
    try:
        current = read_process(proc, item['pid'])
        if any(current[key] != item[key] for key in ('args', 'parent', 'started', 'cgroup')):
            raise RuntimeError(f'Process identity changed for PID {item["pid"]}; not stopped')
        signal.pidfd_send_signal(fd, signal.SIGINT)
    except (FileNotFoundError, ProcessLookupError):
        pass
    finally:
        os.close(fd)


def stop_standard_teleop(proc='/proc', timeout=5.):
    found = standard_teleop(proc)
    for item in found:
        interrupt_verified(item, proc)
    end = time.monotonic()+timeout
    while True:
        remaining = standard_teleop(proc)
        if not remaining or time.monotonic() >= end:
            break
        time.sleep(.1)
    return dict(stopped=[dict(pid=item['pid'], role=item['role']) for item in found],
                remaining=[dict(pid=item['pid'], role=item['role']) for item in remaining])


def stop_robot_teleop(host, user='administrator', container='docker.humble.robot.service'):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:-]*', host):
        raise ValueError('Invalid HSR SSH host')
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.-]*', user):
        raise ValueError('Invalid HSR SSH user')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', container):
        raise ValueError('Invalid HSR container name')
    remote = shlex.join(['docker', 'exec', '-i', '--user', 'root', container, 'python3', '-'])
    command = ['ssh', '-o', 'ConnectTimeout=5', '-o', 'StrictHostKeyChecking=accept-new',
               '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=2',
               '-o', 'NumberOfPasswordPrompts=1', f'{user}@{host}', remote]
    env = dict(os.environ)
    # Match docker/start.sh's existing robot login default, without putting the
    # password in command arguments or logs. SSH keys are still preferred by SSH.
    password = env.get('HSR_SSH_PASSWORD', 'password')
    if password:
        if shutil.which('sshpass') is None:
            raise RuntimeError('sshpass is required for password login; use the HSR container or HSR_SSH_PASSWORD="" with SSH keys')
        command = ['sshpass', '-e', *command]
        env['SSHPASS'] = password
    else:
        command[1:1] = ['-o', 'BatchMode=yes']
    try:
        result = subprocess.run(command, input=Path(__file__).resolve().read_text(),
                                capture_output=True, text=True, timeout=20, env=env)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f'HSR teleop stop timed out: {user}@{host}; PC teleop was not started') from exc
    if result.returncode:
        raise RuntimeError(f'HSR teleop stop failed ({result.returncode}) at {user}@{host}: '
                           f'{result.stderr.strip() or result.stdout.strip()}')
    report = json.loads(result.stdout)
    if not isinstance(report, dict) or report.get('remaining') != [] or not isinstance(report.get('stopped'), list):
        raise RuntimeError(f'HSR teleop did not stop: {report}')
    return dict(host=host, container=container, **report)


if __name__ == '__main__':
    report = stop_standard_teleop()
    print(json.dumps(report), flush=True)
    raise SystemExit(1 if report['remaining'] else 0)
