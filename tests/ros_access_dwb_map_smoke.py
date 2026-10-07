"""Isolated DWB on the real map with laser returns deliberately missing walls."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import rclpy
import yaml

from ros_access_controller_smoke import Plant, spin_for
from ros_access_return_smoke import MapPlant
from ros_access_end_to_end_smoke import run_while_spinning

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import dwvp_access_experiment as experiment
from dwvp_access_path import load_map,assert_clear,PathBlockedError


class MissingWallPlant(MapPlant):
    # Intentionally uninformative scan: the saved map must still protect walls.
    publish_scan=Plant.publish_scan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-session',type=Path,required=True)
    p.add_argument('--map',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sensor-only',action='store_true',help='Control case using the original frozen template')
    p.add_argument('--dwb-horizon',type=float,help='Offline tuning experiment only')
    p.add_argument('--standard-rollout',action='store_true',help='Offline acceleration-ramp experiment only')
    p.add_argument('--raycast-scan',action='store_true',help='Also verify normal wall returns from the map')
    args=p.parse_args()
    assert Path('/.dockerenv').exists() and set(os.listdir('/sys/class/net'))=={'lo'}
    assert os.environ.get('ROS_LOCALHOST_ONLY')=='1'
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    source=args.source_session;original=json.loads((source/'manifest.json').read_text())
    trial=next(t for t in original['trials'] if t['controller']=='DWB')
    _,_,reference=experiment.load_trial(source,trial['id'])
    start=experiment.start_pose(original,trial)
    config=yaml.safe_load((source/'experiment_config.yaml').read_text())
    config['conditions']['E2_environment'].update(methods=['DWB'],start_pose=start.tolist())
    settings=out/'config.yaml';settings.write_text(yaml.safe_dump(config))
    route=out/'route.csv';np.savetxt(route,reference,delimiter=',',header='x,y,yaw',comments='')
    template=source/'params_template.yaml' if args.sensor_only else ROOT/'params/hsrb_dwvp_access_params.yaml'
    if args.dwb_horizon is not None or args.standard_rollout:
        tuned=yaml.safe_load(template.read_text())
        dwb=tuned['controller_server']['ros__parameters']['DWB']
        if args.dwb_horizon is not None:
            dwb['sim_time']=args.dwb_horizon
        if args.standard_rollout:
            dwb.update(trajectory_generator_name='dwb_plugins::StandardTrajectoryGenerator',
                       discretize_by_time=True,time_granularity=1./30.,limit_vel_cmd_in_traj=True)
        template=out/'tuning_template.yaml';template.write_text(yaml.safe_dump(tuned))
    session=out/'session'
    manifest=experiment.prepare(session,template,origin=[0,0,0],environment_path=route,config_file=settings,
        conditions=['E2_environment'],repeats=[1])
    before=yaml.safe_load((source/trial['params_file']).read_text())
    after=yaml.safe_load((session/manifest['trials'][0]['params_file']).read_text())
    for key in ('controller_server','velocity_smoother','amcl','global_costmap'):
        left,right=before[key].copy(),after[key].copy()
        if key=='controller_server' and not args.sensor_only:
            left=dict(left,ros__parameters={k:v for k,v in left['ros__parameters'].items() if k!='DWB'})
            right=dict(right,ros__parameters={k:v for k,v in right['ros__parameters'].items() if k!='DWB'})
        assert left==right,f'Unrelated settings changed: {key}'
    rclpy.init();plant=(MapPlant if args.raycast_scan else MissingWallPlant)(load_map(args.map),start)
    error=None
    try:
        spin_for(plant,1.)
        command=[sys.executable,str(ROOT/'scripts/dwvp_access_batch.py'),'--session',str(session),
                 '--map',str(args.map.resolve()),'--resume','--continue-on-endpoint-failure']
        try:
            run_while_spinning(plant,command,out/'batch.log',230.)
        except AssertionError as exc:
            error=str(exc)
        spin_for(plant,.5)
        trace=np.array(plant.trace)[::5]
        np.savetxt(out/'trace.csv',trace,delimiter=',',header='x,y,yaw',comments='')
        info,_,blocked=load_map(args.map)
        report=dict(physical_trials=0,controller='DWB',sensor_only=args.sensor_only,
            laser_deliberately_missing_walls=not args.raycast_scan,batch_error=error,trace_clear=True,
            dwb_parameters=after['controller_server']['ros__parameters']['DWB'])
        try:
            assert_clear(trace,info,blocked,.22)
        except PathBlockedError as exc:
            report.update(trace_clear=False,first_map_overlap=exc.details)
        result=session/'runs'/manifest['trials'][0]['id']/'result.json'
        report['trial_result']=json.loads(result.read_text()) if result.exists() else None
        if result.exists():
            assert (result.parent/'scan_ranges.csv').stat().st_size>100
            assert (result.parent/'odom_pose.csv').stat().st_size>100
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
        if not args.sensor_only:
            assert error is None, error
            assert report['trace_clear'], report
            assert report['trial_result']['success'], report
            print('PASS: DWB trial and return with static walls; raycast scan=' + str(args.raycast_scan),flush=True)
    finally:
        plant.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':
    main()
