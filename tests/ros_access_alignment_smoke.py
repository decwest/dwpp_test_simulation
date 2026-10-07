"""Isolated Nav2: identical S->G trials, unscored returns, and stopped correction."""
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


class ResidualPlant(Plant):
    def __init__(self):
        super().__init__()
        self.return_target = None
        self.injected = False
        self.create_subscription(RosPath, '/dwvp_access/return_path', self.return_path,
                                 QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))

    def return_path(self, message):
        self.return_target = message.poses[-1].pose if message.poses else None

    def raw_callback(self, command):
        super().raw_callback(command)
        target = self.return_target
        if (not self.injected and target is not None
                and command.linear.x == command.linear.y == command.angular.z == 0.
                and np.hypot(self.x-target.position.x,self.y-target.position.y)<.055):
            # Deliberately reproduce a fresh stopped localization residual after
            # Nav2's success. This tests recovery, not physical robot performance.
            self.x = target.position.x+.1062
            self.y = target.position.y
            self.injected = True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert Path('/.dockerenv').exists() and set(os.listdir('/sys/class/net'))=={'lo'}
    assert os.environ.get('ROS_LOCALHOST_ONLY')=='1'
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    config=experiment.default_config()
    config['conditions']['E2_environment']['methods']=['DWVP']
    settings=out/'config.yaml';settings.write_text(yaml.safe_dump(config))
    route=out/'route.csv'
    x=np.linspace(0,.8,41)
    np.savetxt(route,np.c_[x,np.zeros_like(x),np.zeros_like(x)],delimiter=',',header='x,y,yaw',comments='')
    Image.fromarray(np.full((200,200),254,dtype=np.uint8)).save(out/'map.pgm')
    map_file=out/'map.yaml'
    map_file.write_text(yaml.safe_dump(dict(image='map.pgm',resolution=.05,origin=[-5.,-5.,0.],
        mode='trinary',negate=0,occupied_thresh=.65,free_thresh=.196)))
    session=out/'session'
    manifest=experiment.prepare(session,ROOT/'params/hsrb_dwvp_access_params.yaml',origin=[0,0,0],
        environment_path=route,config_file=settings,conditions=['E2_environment'],repeats=[1,2])
    original={str(p.relative_to(session)):p.read_bytes() for p in session.rglob('*') if p.is_file()}
    command=[sys.executable,str(ROOT/'scripts/dwvp_access_batch.py'),'--session',str(session),
             '--map',str(map_file),'--resume','--continue-on-endpoint-failure']
    rclpy.init();plant=ResidualPlant()
    try:
        spin_for(plant,1.)
        run_while_spinning(plant,command,out/'batch.log',180.)
        assert plant.injected,'Stopped residual was not injected'
        batch_folder=next((session/'batches').iterdir())
        status=json.loads((batch_folder/'status.json').read_text())
        assert status['status']=='completed' and len(status['completed'])==2,status
        assert len(status['returns'])==2 and status['reference_policy']=='session_fixed',status
        checks=[json.loads(p.read_text()) for p in (batch_folder/'alignment_checks').glob('*.json')]
        corrected=[c for c in checks if len(c['attempts'])>1]
        assert len(corrected)==1 and corrected[0]['verified'] and not corrected[0]['attempts'][0]['result_success'],checks
        returns=[json.loads(p.read_text()) for p in (batch_folder/'returns').glob('*/result.json')]
        assert len(returns)==3 and sum(not r['success'] for r in returns)==1,returns
        assert all(r['goal_checker_id']==experiment.ALIGNMENT_GOAL_CHECKER for r in returns)
        for trial in manifest['trials']:
            folder=session/'runs'/trial['id']
            result=json.loads((folder/'result.json').read_text())
            assert result['success'] and result['direction']=='forward',result
            assert result['goal_checker_id']=='general_goal_checker',result
            assert (folder/'reference.csv').read_bytes()==(session/'paths/E2_environment.csv').read_bytes()
            runtime=yaml.safe_load((folder/'controller_server_runtime.yaml').read_text())
            server=runtime['/controller_server']['ros__parameters']
            assert server['general_goal_checker.xy_goal_tolerance']==.1
        assert all((session/name).read_bytes()==data for name,data in original.items())
        spin_for(plant,1.)
        assert np.hypot(plant.x,plant.y)<=.1 and abs(plant.yaw)<=.3
        (out/'report.json').write_text(json.dumps(dict(physical_trials=0,synthetic_trials=2,
            all_trials_same_forward_reference=True,unscored_returns=2,including_last_return=True,
            injected_stopped_residual_m=.1062,correction_verified=True,failed_transfer_retained=True,
            scored_goal_tolerance_unchanged=True,original_frozen_files_unchanged=True),indent=2)+'\n')
        print('PASS: two identical S->G trials, return after each, and correction of a stopped 0.1062 m miss.',flush=True)
    finally:
        plant.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':
    main()
