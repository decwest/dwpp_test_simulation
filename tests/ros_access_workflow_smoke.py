"""Real SLAM/map saver/AMCL/workflow integration in Docker --network none only."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import LaserScan
from tf2_ros import StaticTransformBroadcaster
import yaml

from ros_access_controller_smoke import Plant, spin_for

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import dwvp_access_experiment as experiment
import dwvp_access_workflow as workflow


class RoomPlant(Plant):
    def __init__(self):
        super().__init__()
        # SLAM and AMCL, not the plant, own map and map->odom in this test.
        self.destroy_publisher(self.map_pub)
        self.destroy_publisher(self.static_tf.pub_tf)
        self.static_tf = StaticTransformBroadcaster(self)
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = 'base_link'; tf.child_frame_id = 'base_footprint'
        tf.transform.rotation.w = 1.
        self.static_tf.sendTransform(tf)
        self.received_map = None
        self.localized = False
        self.create_subscription(OccupancyGrid, '/map', self.receive_map,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.receive_pose, 10)
        self.initial_pose = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)

    def receive_map(self, message): self.received_map = message
    def receive_pose(self, message): self.localized = True

    def publish_scan(self):
        scan = LaserScan()
        scan.header.frame_id = 'base_link'; scan.header.stamp = self.get_clock().now().to_msg()
        scan.angle_min = -math.pi; scan.angle_max = math.pi
        scan.angle_increment = 2*math.pi/1440
        scan.range_min = .05; scan.range_max = 12.
        distances = []
        for theta in np.linspace(-math.pi, math.pi, 1441):
            dx, dy = math.cos(theta+self.yaw), math.sin(theta+self.yaw)
            tx = ((5.-self.x)/dx if dx>0 else (-3.-self.x)/dx) if abs(dx)>1e-8 else math.inf
            ty = ((3.-self.y)/dy if dy>0 else (-3.-self.y)/dy) if abs(dy)>1e-8 else math.inf
            distances.append(float(min(tx,ty)))
        scan.ranges = distances
        self.scan_pub.publish(scan)


def interactive(plant, command, logfile, ready, timeout):
    master, slave = os.openpty()
    with logfile.open('w') as log:
        process = subprocess.Popen(command, stdin=slave, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        os.close(slave)
        entered = False
        end = time.monotonic()+timeout
        try:
            while process.poll() is None:
                rclpy.spin_once(plant, timeout_sec=.02)
                if not entered and ready():
                    os.write(master, b'\n'); entered = True
                assert time.monotonic()<end, logfile.read_text()
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                stop = time.monotonic()+45
                while process.poll() is None and time.monotonic()<stop:
                    rclpy.spin_once(plant, timeout_sec=.02)
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            os.close(master)
        assert entered and process.returncode == 0, logfile.read_text()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--one-way', action='store_true', help='Exercise two ordered filmed response conditions and returns')
    parser.add_argument('--stationary-noise', type=float, help='Record noise before preparing the synthetic trials')
    args = parser.parse_args()
    assert Path('/.dockerenv').exists() and set(os.listdir('/sys/class/net')) == {'lo'}
    assert os.environ.get('ROS_LOCALHOST_ONLY') == '1'
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    workspace = out/'workspace'; workspace.mkdir()
    command = [sys.executable, str(ROOT/'scripts/dwvp_access_workflow.py'), '--workspace', str(workspace)]
    config = experiment.default_config()
    conditions = list(experiment.RESPONSE_CONDITIONS[:2]) if args.one_way else ['E1_lateral']
    for name in conditions:
        config['conditions'][name].update(methods=['DWVP'], length_m=.8, start_pose=[0.,.2,0.])
        config['conditions'][name]['evaluation']['goal_margin_m'] = .1
    settings = out/'config.yaml'; settings.write_text(yaml.safe_dump(config))
    rclpy.init(); plant = RoomPlant()
    try:
        spin_for(plant, 1.)
        interactive(plant, command+['mapping', '--no-rviz', '--no-joy'], out/'mapping.log',
                    lambda: plant.received_map is not None and len(plant.received_map.data)>100, 60)
        saved = workflow.choose_map(workspace)
        assert saved.parent.name.startswith('lab_')
        assert workflow.checked_map(saved)['free_thresh'] == .196
        assert json.loads((saved.parent/'workflow.json').read_text())['status']=='completed'
        assert plant.raw_count == 0
        print(f'PASS: actual SLAM Toolbox and map saver, timestamped map, no motion: {saved}', flush=True)
        spin_for(plant, 3.)
        sent_pose = None
        def ready():
            nonlocal sent_pose
            if sent_pose is None and plant.initial_pose.get_subscription_count()>0:
                pose = PoseWithCovarianceStamped(); pose.header.frame_id='map'
                pose.header.stamp=plant.get_clock().now().to_msg()
                pose.pose.pose.orientation.w=1.
                pose.pose.covariance[0]=pose.pose.covariance[7]=.0025
                pose.pose.covariance[35]=.0009
                plant.initial_pose.publish(pose); sent_pose=time.monotonic()
            return plant.localized and sent_pose is not None and time.monotonic()-sent_pose>2
        options = ['--one-way', '--ordered'] if args.one_way else []
        if args.stationary_noise is not None:
            options += ['--stationary-noise', str(args.stationary_noise)]
        interactive(plant, command+['experiment', '--no-rviz', '--no-joy', '--conditions', *conditions,
                    '--repeats', '1', '--config', str(settings)]+options, out/'experiment.log', ready, 240)
        pointer = json.loads((workspace/'results/dwvp_access/latest.json').read_text())
        session = workspace/pointer['session']
        manifest = json.loads((session/'manifest.json').read_text())
        assert len(manifest['trials']) == len(conditions)
        assert bool(manifest.get('bidirectional')) is not args.one_way
        assert bool(manifest.get('ordered')) is args.one_way
        for trial in manifest['trials']:
            result = json.loads((session/'runs'/trial['id']/'result.json').read_text())
            assert result['success']
            assert result['reference_policy'] == ('session_fixed' if args.one_way else 'per_trial_current_pose')
            if args.one_way:
                _, _, reference = experiment.load_trial(session, trial['id'])
                np.testing.assert_allclose(np.loadtxt(session/'runs'/trial['id']/'reference.csv',
                    delimiter=',', skiprows=1), reference)
                np.testing.assert_allclose(experiment.start_pose(manifest, trial), manifest['measured_start'])
        if args.one_way:
            batch = next((session/'batches').iterdir())
            status = json.loads((batch/'status.json').read_text())
            assert status['status'] == 'completed' and len(status['returns']) == len(conditions)
            assert status['reference_policy'] == 'session_fixed'
        if args.stationary_noise is not None:
            noise = json.loads((session.parent/'stationary_noise/noise_summary.json').read_text())
            assert noise['stationary_verified'] and noise['samples'] > 1 and noise['frame'] == 'map'
            assert noise['requested_duration_s'] == args.stationary_noise
            log = (out/'experiment.log').read_text()
            assert log.index('stationary-noise') < log.index(' prepare ') < log.index('dwvp_access_batch.py')
        assert json.loads((session.parent/'workflow.json').read_text())['status']=='completed'
        assert (session.parent/'map/map.pgm').read_bytes() == saved.with_suffix('.pgm').read_bytes()
        spin_for(plant, 3.)
        assert not plant.get_publishers_info_by_topic('/omni_base_controller/cmd_vel')
        assert not plant.get_publishers_info_by_topic('/map')
        (out/'report.json').write_text(json.dumps(dict(physical_trials=0, synthetic_trials=len(conditions),
            one_way=args.one_way, stationary_noise_seconds=args.stationary_noise,
            real_slam_and_map_saver=True, real_amcl_and_controller=True,
            timestamped_map=str(saved), timestamped_session=str(session), owned_processes_stopped=True), indent=2)+'\n')
        print('PASS: newest map -> actual AMCL -> new session -> batch -> summary and cleanup', flush=True)
    finally:
        plant.destroy_node(); rclpy.try_shutdown()


if __name__=='__main__':
    main()
