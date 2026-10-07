"""Real Nav2 batch with an injected post-goal pose residual; isolated Docker only."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import rclpy
from nav_msgs.msg import Path as RosPath
from rclpy.qos import DurabilityPolicy, QoSProfile
import yaml

from ros_access_controller_smoke import Plant, spin_for
from ros_access_end_to_end_smoke import run_while_spinning

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import dwvp_access_experiment as experiment
from dwvp_access_batch import settled_endpoint_failure


class ResidualPlant(Plant):
    def __init__(self):
        super().__init__()
        self.reference = None
        self.injected = False
        self.create_subscription(RosPath, '/dwvp_access/reference_path', self.reference_callback,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

    def reference_callback(self, message):
        self.reference = message.poses[-1].pose

    def raw_callback(self, command):
        super().raw_callback(command)
        if (not self.injected and self.reference is not None
                and command.linear.x == command.linear.y == command.angular.z == 0.
                and np.hypot(self.x-self.reference.position.x, self.y-self.reference.position.y) < .11):
            # Exercise batch failure handling, not physical/controller performance:
            # a fresh final TF outside yaw tolerance after Nav2 already succeeded.
            self.yaw += .52
            self.injected = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert Path('/.dockerenv').exists() and set(os.listdir('/sys/class/net')) == {'lo'}
    assert os.environ.get('ROS_LOCALHOST_ONLY') == '1'
    out = args.output; out.mkdir(parents=True, exist_ok=False)
    config = experiment.default_config()
    config['conditions']['E1_lateral'].update(length_m=.8, start_pose=[0., .2, 0.])
    config['conditions']['E1_lateral']['evaluation']['goal_margin_m'] = .1
    settings = out/'config.yaml'; settings.write_text(yaml.safe_dump(config))
    Image.fromarray(np.full((200,200),254,dtype=np.uint8)).save(out/'map.pgm')
    map_file = out/'map.yaml'
    map_file.write_text(yaml.safe_dump(dict(image='map.pgm', resolution=.05, origin=[-5.,-5.,0.],
                                          mode='trinary', negate=0, occupied_thresh=.65, free_thresh=.196)))
    session = out/'session'
    manifest = experiment.prepare(session, ROOT/'params/hsrb_dwvp_access_params.yaml',
        current_start=[0,0,0], config_file=settings, bidirectional=True,
        conditions=['E1_lateral'], repeats=[1,2])
    command = [sys.executable, str(ROOT/'scripts/dwvp_access_batch.py'), '--session', str(session),
               '--map', str(map_file), '--start-from-current', '--continue-on-endpoint-failure']
    rclpy.init(); plant = ResidualPlant()
    try:
        plant.reset([0.,0.,0.]); spin_for(plant, 1.)
        # Simulate resuming after a previous batch hit the injected endpoint miss:
        # first run only repetition 1 with the original stop-on-failure behavior.
        try:
            run_while_spinning(plant, command[:-1]+['--repeats','1'], out/'first_batch.log', 120.)
        except (RuntimeError, AssertionError):
            pass
        else:
            raise AssertionError('Strict batch must stop on the injected endpoint miss')
        first = session/'runs'/manifest['trials'][0]['id']
        result = json.loads((first/'result.json').read_text())
        assert plant.injected and settled_endpoint_failure(result, manifest, first), result
        assert len(list((session/'runs').iterdir())) == 1
        preserved = {str(p): p.read_bytes() for p in first.rglob('*') if p.is_file()}
        # A second injected residual exercises live continuation as well as resume.
        plant.injected = False
        plant.reference = None
        run_while_spinning(plant, command+['--resume'], out/'resumed_batch.log', 240.)
        assert all(Path(p).read_bytes() == contents for p, contents in preserved.items())
        report = experiment.summarize(session)
        assert report['recorded'] == 4 and report['pending'] == 0, report
        assert not report['trials'][0]['success']
        results = [json.loads((session/'runs'/t['id']/'result.json').read_text()) for t in manifest['trials']]
        assert results[1]['success'] is False and results[2]['success'] and results[3]['success'], results
        count = plant.raw_count
        run_while_spinning(plant, command+['--resume'], out/'completed_resume.log', 20.)
        assert count == plant.raw_count
        (out/'report.json').write_text(json.dumps(dict(physical_trials=0, recorded=4, failed=2,
            succeeded=2, original_attempt_unchanged=True, live_continuation=True,
            completed_resume_no_goals=True), indent=2)+'\n')
        print('PASS: stopped endpoint failures remain failed; resume and live continuation complete remaining legs.')
    finally:
        plant.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()
