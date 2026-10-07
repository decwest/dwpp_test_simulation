"""Only identified standard robot teleop may be stopped during workflow startup."""
import json
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import dwvp_access_robot_teleop as teleop
import dwvp_access_workflow as workflow


def process(proc, pid, argv, parent=1, group='0::/robot\n', started=100):
    folder = proc/str(pid); folder.mkdir(exist_ok=True)
    (folder/'cmdline').write_bytes(('\0'.join(argv)+'\0').encode())
    (folder/'stat').write_text(f'{pid} (process with spaces) '+' '.join(['S', str(parent), *['0']*17, str(started)]))
    (folder/'cgroup').write_text(group)


def robot_proc(tmp_path):
    proc = tmp_path/'proc'; proc.mkdir()
    (proc/'self').mkdir(); (proc/'self/cgroup').write_text('0::/robot\n')
    process(proc, 10, ['/usr/bin/python3', '/opt/ros/humble/bin/ros2', 'launch', 'hsrb_robot_launch', 'boot_app.launch.py'])
    process(proc, 11, [teleop.JOY_DRIVER, '--ros-args'], parent=10)
    process(proc, 12, ['/usr/bin/python3', teleop.JOY_CONTROL, '--ros-args'], parent=10)
    return proc


def test_only_standard_children_in_selected_container_match(tmp_path):
    proc = robot_proc(tmp_path)
    process(proc, 20, ['/usr/bin/python3', 'robot_service'])
    process(proc, 21, [teleop.JOY_DRIVER], parent=20)  # User-managed launch.
    process(proc, 22, ['/usr/bin/python3', teleop.JOY_CONTROL], parent=10, group='0::/other\n')
    process(proc, 23, ['/usr/bin/python3', '/tmp/joystick_control_node'], parent=10)
    process(proc, 24, ['echo', teleop.JOY_CONTROL], parent=10)
    process(proc, 25, ['/root/ros2_ws/install/hsrb_bringup/driver'], parent=10)
    assert [(p['pid'],p['role']) for p in teleop.standard_teleop(proc)] == [(11,'joy_driver'),(12,'joystick_control')]


def test_graceful_stop_idempotence_and_preserved_parent(monkeypatch, tmp_path):
    proc = robot_proc(tmp_path)
    signals = []
    def interrupt(item, root):
        signals.append(item['pid'])
        for file in (root/str(item['pid'])).iterdir(): file.unlink()
        (root/str(item['pid'])).rmdir()
    monkeypatch.setattr(teleop, 'interrupt_verified', interrupt)
    report = teleop.stop_standard_teleop(proc)
    assert signals == [11,12] and report['remaining'] == []
    assert (proc/'10/cmdline').is_file()
    assert teleop.stop_standard_teleop(proc) == dict(stopped=[],remaining=[])


def test_non_exiting_or_respawned_teleop_is_reported(monkeypatch, tmp_path):
    proc = robot_proc(tmp_path)
    monkeypatch.setattr(teleop, 'interrupt_verified', lambda *args: None)
    report = teleop.stop_standard_teleop(proc, timeout=0)
    assert [p['pid'] for p in report['remaining']] == [11,12]


@pytest.mark.parametrize('recycled', [False, True])
def test_pid_identity_checked_before_signal(monkeypatch, tmp_path, recycled):
    proc = robot_proc(tmp_path)
    target = teleop.standard_teleop(proc)[0]
    calls = []
    monkeypatch.setattr(teleop.os, 'pidfd_open', lambda pid: 900)
    monkeypatch.setattr(teleop.os, 'close', lambda fd: calls.append(('close',fd)))
    monkeypatch.setattr(teleop.signal, 'pidfd_send_signal', lambda fd,sig: calls.append((fd,sig)))
    if recycled:
        process(proc, 11, [teleop.JOY_DRIVER, '--ros-args'], parent=10, started=200)
        with pytest.raises(RuntimeError, match='identity changed'):
            teleop.interrupt_verified(target, proc)
        assert calls == [('close',900)]
    else:
        teleop.interrupt_verified(target, proc)
        assert calls == [(900,signal.SIGINT),('close',900)]


def test_ssh_keeps_password_out_of_command_and_loggable_report(monkeypatch):
    password = 'test-password-only'
    monkeypatch.setenv('HSR_SSH_PASSWORD', password)
    monkeypatch.setattr(teleop.shutil, 'which', lambda _: '/usr/bin/sshpass')
    def run(command, **kwargs):
        assert password not in ' '.join(command)
        assert kwargs['env']['SSHPASS'] == password
        assert 'docker exec -i --user root docker.humble.robot.service python3 -' == command[-1]
        assert 'def stop_standard_teleop' in kwargs['input']
        return NS(returncode=0, stdout='{"stopped": [], "remaining": []}', stderr='')
    monkeypatch.setattr(teleop.subprocess, 'run', run)
    report = teleop.stop_robot_teleop('192.168.50.10')
    assert report['remaining'] == [] and password not in json.dumps(report)


@pytest.mark.parametrize('mode', ['ssh_failure','timeout','still_running'])
def test_ssh_or_stop_failure_cannot_be_accepted(monkeypatch, mode):
    monkeypatch.setenv('HSR_SSH_PASSWORD', '')
    def run(command, **kwargs):
        assert 'BatchMode=yes' in command
        if mode=='timeout': raise subprocess.TimeoutExpired(command,20)
        return NS(returncode=255 if mode=='ssh_failure' else 0,
                  stdout='{"stopped": [], "remaining": [1]}', stderr='connection failed')
    monkeypatch.setattr(teleop.subprocess, 'run', run)
    with pytest.raises(RuntimeError): teleop.stop_robot_teleop('192.168.50.10')


@pytest.mark.parametrize('kwargs', [dict(host='-bad'), dict(host='host; true'),
                                  dict(host='host',user='root; true'), dict(host='host',container='x; true')])
def test_connection_arguments_cannot_inject_shell_commands(kwargs):
    with pytest.raises(ValueError): teleop.stop_robot_teleop(**kwargs)


class Graph:
    def __init__(self, publishers=None, nodes=()):
        self.publishers, self.nodes = publishers or {}, nodes
    def get_node_names_and_namespaces(self): return [(name,'/') for name in self.nodes]
    def get_publishers_info_by_topic(self, topic):
        return [NS(node_name=name,node_namespace='/') for name in self.publishers.get(topic,[])]


def test_only_joy_conflicts_are_eligible_for_handoff():
    graph = Graph({'/omni_base_controller/cmd_vel': ['joystick_control_node']*2, '/joy':['joy_node']})
    assert workflow.inspect_idle(graph, allow_robot_joy=True)
    with pytest.raises(RuntimeError): workflow.inspect_idle(graph)
    assert not workflow.inspect_idle(Graph(), allow_robot_joy=True)


@pytest.mark.parametrize('graph', [Graph({'/map':['map_server']}),
    Graph({'/omni_base_controller/cmd_vel':['joystick_control_node','teleop_twist_keyboard']}),
    Graph({'/joy':['unknown_input']}), Graph({'/cmd_vel_nav':['controller_server']}),
    Graph(nodes=['slam_toolbox']), Graph(nodes=['controller_server'])])
def test_other_active_nodes_prevent_handoff(graph):
    with pytest.raises(RuntimeError): workflow.inspect_idle(graph, allow_robot_joy=True)


def test_dry_run_never_performs_remote_handoff(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'argv', ['workflow', '--workspace', str(tmp_path), 'mapping', '--dry-run'])
    monkeypatch.setattr(workflow, 'require_idle', lambda *args: pytest.fail('Dry-run must not start ROS/SSH'))
    workflow.main()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('mode', ['stopped','clock_offset','moving','stalled_stamp','missing','nonfinite'])
def test_handoff_requires_stationary_progressing_odometry(monkeypatch, mode):
    import rclpy
    elapsed = [0.]
    class Observer:
        callback = None
        destroyed = False
        def create_subscription(self, kind, topic, callback, qos):
            self.callback = callback
            return object()
        def destroy_subscription(self, subscription): self.destroyed = True
        def get_clock(self): return NS(now=lambda: NS(nanoseconds=int((10+elapsed[0])*1e9)))
    node = Observer()
    def spin(node, timeout_sec):
        elapsed[0] += .02
        if mode=='missing': return
        stamp = 10.+elapsed[0]+(1.55 if mode=='clock_offset' else 0.)
        if mode=='stalled_stamp': stamp=10.
        x = .1 if mode=='moving' else (float('nan') if mode=='nonfinite' else 0.)
        msg = NS(header=NS(stamp=NS(sec=int(stamp),nanosec=int((stamp-int(stamp))*1e9))),
                 twist=NS(twist=NS(linear=NS(x=x,y=0.), angular=NS(z=0.))))
        node.callback(msg)
    monkeypatch.setattr(workflow.time, 'monotonic', lambda: elapsed[0])
    monkeypatch.setattr(rclpy, 'spin_once', spin)
    if mode in ('stopped','clock_offset'):
        report=workflow.stopped_wheels(node, timeout=.8)
        assert report['source_clock_synchronized'] == (mode=='stopped')
        assert elapsed[0]>=.5 and report['stationary_interval_s']==.5
    else:
        with pytest.raises(RuntimeError,match='Release LB'):
            workflow.stopped_wheels(node, timeout=.8)
    assert node.destroyed
