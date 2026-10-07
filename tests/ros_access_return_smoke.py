"""Isolated Nav2 returns on a recorded E2 map; source experiment is read-only."""
import argparse
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
import yaml

from ros_access_controller_smoke import Plant, spin_for
from ros_access_end_to_end_smoke import run_while_spinning

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import dwvp_access_experiment as experiment
import dwvp_access_batch as batch
from dwvp_access_path import load_map, assert_clear


class MapPlant(Plant):
    def __init__(self, data, pose):
        self.info, self.pixels, self.blocked = data
        self.trace = []
        super().__init__()
        self.reset(pose)
        grid = OccupancyGrid()
        grid.header.frame_id = 'map'
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.info.resolution = self.info['resolution']
        grid.info.height, grid.info.width = self.pixels.shape
        x,y,angle = self.info['origin']
        assert angle == 0., 'Synthetic ray caster requires an axis-aligned map'
        grid.info.origin.position.x,grid.info.origin.position.y = x,y
        grid.info.origin.orientation.w = 1.
        occupancy = (255-self.pixels)/255.
        values = np.where(occupancy > self.info['occupied_thresh'],100,
                          np.where(occupancy < self.info['free_thresh'],0,-1))
        grid.data = np.flipud(values).ravel().tolist()
        self.map_pub.publish(grid)

    def tick(self):
        super().tick()
        self.trace.append([self.x,self.y,self.yaw])

    def publish_scan(self):
        scan = LaserScan()
        scan.header.frame_id = 'base_link'
        scan.header.stamp = self.get_clock().now().to_msg()
        scan.angle_min,scan.angle_max = -math.pi,math.pi
        scan.angle_increment = 2*math.pi/180
        scan.range_min,scan.range_max = .05,5.
        angles = self.yaw+np.linspace(-math.pi,math.pi,181)
        distances = np.arange(.05,5.,self.info['resolution']/2)
        xy = np.array([self.x,self.y])+distances[None,:,None]*np.c_[np.cos(angles),np.sin(angles)][:,None,:]
        indices = np.floor((xy-np.array(self.info['origin'][:2]))/self.info['resolution']).astype(int)
        gx,gy = indices[:,:,0],indices[:,:,1]
        h,w = self.blocked.shape
        inside = (gx>=0)&(gx<w)&(gy>=0)&(gy<h)
        hit = ~inside | self.blocked[np.clip(gy,0,h-1),np.clip(gx,0,w-1)]
        ranges = np.where(hit.any(axis=1),distances[hit.argmax(axis=1)],float('inf'))
        scan.ranges = ranges.tolist()
        self.scan_pub.publish(scan)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-session',type=Path,required=True)
    parser.add_argument('--map',type=Path,required=True)
    parser.add_argument('--stopped-check',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert Path('/.dockerenv').exists() and set(os.listdir('/sys/class/net'))=={'lo'}
    assert os.environ.get('ROS_LOCALHOST_ONLY')=='1'
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    source=args.source_session
    manifest=json.loads((source/'manifest.json').read_text())
    first=manifest['trials'][0]
    _,_,reference=experiment.load_trial(source,first['id'])
    start=experiment.start_pose(manifest,first)
    stopped=json.loads(args.stopped_check.read_text())['initial']['pose']
    data=load_map(args.map)
    radius=.22
    def unused(*args):
        raise AssertionError('A clear E2 corridor must not call NavFn')
    route=batch.checked_return_path(stopped,start,data,radius,unused,out/'actual_stop_return_check.json',reference=reference)
    np.savetxt(out/'actual_stop_return.csv',route,delimiter=',',header='x,y,yaw',comments='')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    info,pixels,_=data;h,w=pixels.shape;x,y,_=info['origin'];res=info['resolution']
    fig,ax=plt.subplots(figsize=(10,6))
    ax.imshow(pixels,cmap='gray',vmin=0,vmax=255,extent=[x,x+w*res,y,y+h*res])
    ax.plot(reference[:,0],reference[:,1],color='tab:blue',lw=4,alpha=.4,label='Frozen scored S->G')
    ax.plot(route[:,0],route[:,1],'--',color='tab:green',label='Checked unscored return')
    ax.scatter([2.440],[-1.590],color='red',marker='x',label='Previous return rejection')
    ax.scatter([stopped[0],start[0]],[stopped[1],start[1]],color=['orange','green'])
    ax.set_aspect('equal');ax.set_xlabel('map x [m]');ax.set_ylabel('map y [m]');ax.legend()
    fig.savefig(out/'return_map.png',dpi=150);plt.close(fig)
    print('PASS: reference return from actual recorded stopped pose clears the actual map.',flush=True)
    config=yaml.safe_load((source/'experiment_config.yaml').read_text())
    config['conditions']['E2_environment']['methods']=['DWVP']
    config['conditions']['E2_environment']['start_pose']=start.tolist()
    settings=out/'config.yaml';settings.write_text(yaml.safe_dump(config))
    path_file=out/'route.csv'
    np.savetxt(path_file,reference,delimiter=',',header='x,y,yaw',comments='')
    session=out/'session'
    test_manifest=experiment.prepare(session,ROOT/'params/hsrb_dwvp_access_params.yaml',origin=[0,0,0],
        environment_path=path_file,config_file=settings,conditions=['E2_environment'],repeats=[1,2])
    original={str(p.relative_to(session)):p.read_bytes() for p in session.rglob('*') if p.is_file()}
    command=[sys.executable,str(ROOT/'scripts/dwvp_access_batch.py'),'--session',str(session),
             '--map',str(args.map.resolve()),'--resume','--continue-on-endpoint-failure']
    rclpy.init();plant=MapPlant(data,stopped)
    try:
        spin_for(plant,1.)
        run_while_spinning(plant,command,out/'batch.log',420.)
        folder=next((session/'batches').iterdir())
        status=json.loads((folder/'status.json').read_text())
        assert status['status']=='completed' and len(status['completed'])==2,status
        assert len(status['returns'])==2 and len(status['alignments'])==1,status
        checks=[json.loads(p.read_text()) for p in (folder/'return_checks').glob('*.json')]
        assert len(checks)>=3 and all(c['verified'] and c['selected']=='frozen_reference_reverse' for c in checks)
        for trial in test_manifest['trials']:
            result=json.loads((session/'runs'/trial['id']/'result.json').read_text())
            assert result['success'] and result['goal_checker_id']=='general_goal_checker',result
            assert (session/'runs'/trial['id']/'reference.csv').read_bytes()==(session/'paths/E2_environment.csv').read_bytes()
        assert all((session/name).read_bytes()==data for name,data in original.items())
        assert batch.pose_close([plant.x,plant.y,plant.yaw],start,test_manifest)
        trace=np.array(plant.trace)[::5]
        np.savetxt(out/'synthetic_trace.csv',trace,delimiter=',',header='x,y,yaw',comments='')
        assert_clear(trace,info,data[2],radius)
        experiment.write_json(out/'report.json',dict(physical_trials=0,synthetic_trials=2,
            actual_map_checked=True,resume_return_from_recorded_stop=True,
            returns_including_last=2,scored_paths_unchanged=True,
            simulated_trace_clear=True,all_returns_use_frozen_corridor=True))
        print('PASS: resume from goal, two fixed trials and all returns on the recorded obstacle map.',flush=True)
    finally:
        plant.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':
    main()
