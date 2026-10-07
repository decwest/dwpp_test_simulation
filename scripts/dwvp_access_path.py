#!/usr/bin/env python3
"""Generate an E2 NavFn path without hardware, in Docker --network none only.

Save map-coordinate x,y,yaw CSV, preview, planner settings and input hashes.
Position smoothing pins endpoints; every final heading is recomputed forward.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import yaml


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def smooth_positions(xy, settings):
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2 or not np.isfinite(xy).all():
        raise ValueError('At least two finite path positions are required')
    xy = xy[np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1)>1e-9]]
    if len(xy)<2:
        raise ValueError('Path has no length')
    spacing = settings['sample_spacing_m']
    if spacing<=0 or not math.isfinite(spacing):
        raise ValueError('sample_spacing_m must be positive')
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
    sample = np.linspace(0,arc[-1],max(2,int(math.ceil(arc[-1]/spacing))+1))
    original = np.c_[np.interp(sample,arc,xy[:,0]),np.interp(sample,arc,xy[:,1])]
    result = original.copy()
    if settings['method'] not in ('none','elastic'):
        raise ValueError('Smoothing method must be none or elastic')
    if settings['method']=='elastic':
        a,b = settings['data_weight'],settings['smooth_weight']
        if not (a>=0 and b>=0 and a+2*b<=1 and settings['iterations']>=1 and settings['tolerance_m']>0):
            raise ValueError('Require nonnegative weights with data_weight + 2*smooth_weight <= 1')
        for _ in range(settings['iterations']):
            delta = a*(original[1:-1]-result[1:-1])+b*(result[:-2]+result[2:]-2*result[1:-1])
            result[1:-1] += delta
            if not len(delta) or np.max(np.abs(delta)) < settings['tolerance_m']:
                break
    return result


def tangent_path(xy):
    xy = np.asarray(xy,dtype=float)
    segments = np.diff(xy,axis=0)
    if np.any(np.linalg.norm(segments,axis=1)<=1e-9):
        raise ValueError('Duplicate positions after smoothing')
    # Central secants, with forward/backward endpoint differences. Never infer reverse travel from input yaw.
    tangent = np.gradient(xy,axis=0)
    if np.any(np.linalg.norm(tangent,axis=1)<=1e-9):
        raise ValueError('Undefined tangent (cusp) after smoothing')
    yaw = np.arctan2(tangent[:,1],tangent[:,0])
    return np.c_[xy,yaw]


def load_map(file):
    from PIL import Image
    file = Path(file)
    info = yaml.safe_load(file.read_text())
    if info.get('mode','trinary') != 'trinary':
        raise ValueError('Map validation currently supports trinary occupancy maps only')
    pixels = np.asarray(Image.open(file.parent/info['image']).convert('L'))
    occupancy = pixels/255 if info.get('negate',0) else (255-pixels)/255
    # Unknown is blocked. This also rejects cells at either threshold boundary.
    blocked = occupancy >= info['free_thresh']
    return info,pixels,np.flipud(blocked)


class PathBlockedError(ValueError):
    def __init__(self, details):
        self.details = details
        x, y = details['path_point_map']
        super().__init__(f'Path footprint intersects occupied/unknown/outside map at '
                         f'({x:.3f}, {y:.3f}) m; check alignment and path endpoints')


def assert_clear(path, info, blocked, radius):
    resolution = float(info['resolution'])
    if resolution <= 0 or radius < 0:
        raise ValueError('Invalid map resolution or robot radius')
    xy = np.asarray(path)[:,:2]
    # Densify segments so smoothing cannot cut through an obstacle between samples.
    dense=[]
    for a,b in zip(xy[:-1],xy[1:]):
        dense.extend(a+np.linspace(0,1,max(2,int(math.ceil(np.linalg.norm(b-a)/(resolution/2)))+1))[:,None]*(b-a))
    ox,oy,angle = info['origin'];c,s=math.cos(angle),math.sin(angle)
    local=(np.asarray(dense)-[ox,oy])@np.array([[c,-s],[s,c]])
    reach=int(math.ceil(radius/resolution))+1
    height,width=blocked.shape
    for point, point_map in zip(local, dense):
        mx,my=np.floor(point/resolution).astype(int)
        for gy in range(my-reach,my+reach+1):
            for gx in range(mx-reach,mx+reach+1):
                centre=(np.array([gx,gy])+.5)*resolution
                # Exact distance from centre position to the cell rectangle.
                distance=np.linalg.norm(np.maximum(np.abs(point-centre)-resolution/2,0))
                if distance<=radius and (gx<0 or gy<0 or gx>=width or gy>=height or blocked[gy,gx]):
                    cell_map = centre@np.array([[c,s],[-s,c]])+[ox,oy]
                    raise PathBlockedError(dict(path_point_map=point_map.tolist(),
                        cell_map=cell_map.tolist(), cell_index=[int(gx),int(gy)],
                        outside_map=bool(gx<0 or gy<0 or gx>=width or gy>=height),
                        robot_radius_m=float(radius), distance_to_cell_m=float(distance)))


def plot_path(file, path, raw, info, pixels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.transforms import Affine2D
    fig,ax=plt.subplots(figsize=(8,7))
    h,w=pixels.shape;res=info['resolution'];ox,oy,yaw=info['origin']
    transform=Affine2D().rotate(yaw).translate(ox,oy)+ax.transData
    ax.imshow(pixels,cmap='gray',vmin=0,vmax=255,origin='upper',extent=[0,w*res,0,h*res],transform=transform)
    ax.plot(raw[:,0],raw[:,1],color='tab:orange',alpha=.7,label='NavFn positions')
    ax.plot(path[:,0],path[:,1],color='tab:blue',label='Smoothed path')
    step=max(1,len(path)//30)
    ax.quiver(path[::step,0],path[::step,1],np.cos(path[::step,2]),np.sin(path[::step,2]),
              angles='xy',scale_units='xy',scale=6,color='tab:red',width=.004)
    ax.scatter(*path[0,:2],color='green',label='Path start');ax.scatter(*path[-1,:2],color='purple',label='Path end')
    ax.set_xlim(path[:,0].min()-.7,path[:,0].max()+.7);ax.set_ylim(path[:,1].min()-.7,path[:,1].max()+.7)
    ax.set_aspect('equal');ax.set_xlabel('map x [m]');ax.set_ylabel('map y [m]');ax.legend()
    fig.tight_layout();fig.savefig(file,dpi=160);plt.close(fig)


def planner_parameters(map_path, config, resolution):
    p=config['planner']
    return {'map_server':{'ros__parameters':{'yaml_filename':str(map_path)}},
            'planner_server':{'ros__parameters':{'expected_planner_frequency':1.0,'planner_plugins':['NavFn'],
                'NavFn':{'plugin':'nav2_navfn_planner/NavfnPlanner',**{k:p[k] for k in ('tolerance','use_astar','allow_unknown')}}}},
            'global_costmap':{'global_costmap':{'ros__parameters':{'update_frequency':1.0,'publish_frequency':1.0,
                'global_frame':'map','robot_base_frame':'base_link','robot_radius':p['robot_radius_m'],
                'resolution':resolution,'track_unknown_space':True,'plugins':['static_layer','inflation_layer'],
                'static_layer':{'plugin':'nav2_costmap_2d::StaticLayer','map_subscribe_transient_local':True},
                'inflation_layer':{'plugin':'nav2_costmap_2d::InflationLayer','inflation_radius':p['inflation_radius_m'],
                                  'cost_scaling_factor':p['cost_scaling_factor']},'always_send_full_costmap':True}}}}


def generate(args):
    if not Path('/.dockerenv').exists() or set(os.listdir('/sys/class/net')) != {'lo'} or os.environ.get('ROS_LOCALHOST_ONLY') != '1':
        raise RuntimeError('Run in Docker --network none with ROS_LOCALHOST_ONLY=1; no hardware connection is allowed')
    import rclpy
    from ament_index_python.packages import get_package_prefix
    from action_msgs.msg import GoalStatus
    from geometry_msgs.msg import PoseStamped
    from nav2_msgs.action import ComputePathToPose
    from nav_msgs.msg import OccupancyGrid
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile
    import shutil
    if not np.isfinite([*args.start,*args.goal]).all():
        raise ValueError('Finite start and goal required')
    config=yaml.safe_load(args.config.read_text())
    info,pixels,blocked=load_map(args.map)
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(args.map,output/'input_map.yaml')
    image_name = 'input_map_image' + Path(info['image']).suffix
    shutil.copy2(args.map.parent/info['image'],output/image_name)
    local_map=dict(info,image=image_name)
    (output/'map.yaml').write_text(yaml.safe_dump(local_map))
    (output/'path_config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    params=planner_parameters(output/'map.yaml',config,info['resolution'])
    paramfile=output/'planner_params.yaml';paramfile.write_text(yaml.safe_dump(params,sort_keys=False))
    processes=[];logs=[];rclpy.init();node=Node('dwvp_access_path_generator')
    costmaps=[]
    def receive_costmap(msg):
        costmaps[:] = [msg]
    costmap_sub=node.create_subscription(OccupancyGrid,'/global_costmap/costmap',
        receive_costmap,
        QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
    def spawn(package,executable,extra):
        log=(output/(executable+'.log')).open('w');logs.append(log)
        # Own the node process itself so shutdown cannot leave ros2-run children alive.
        binary=Path(get_package_prefix(package))/'lib'/package/executable
        processes.append(subprocess.Popen([str(binary),*extra],stdout=log,stderr=subprocess.STDOUT))
    try:
        spawn('tf2_ros','static_transform_publisher',['--x',str(args.start[0]),'--y',str(args.start[1]),'--yaw',str(args.start[2]),'--frame-id','map','--child-frame-id','base_link'])
        for package,executable in (('nav2_map_server','map_server'),('nav2_planner','planner_server')):
            spawn(package,executable,['--ros-args','--params-file',str(paramfile)])
        spawn('nav2_lifecycle_manager','lifecycle_manager',['--ros-args','-p','autostart:=true','-p',"node_names:=['map_server','planner_server']",'-p','bond_timeout:=0.0'])
        client=ActionClient(node,ComputePathToPose,'/compute_path_to_pose')
        if not client.wait_for_server(timeout_sec=60):
            raise RuntimeError('NavFn action unavailable; see saved node logs')
        # Action availability can precede the first static-layer update. Plan only
        # after a published, initialized costmap matches the map loaded above.
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            rclpy.spin_once(node,timeout_sec=.1)
            if (costmaps and costmaps[0].header.frame_id=='map'
                    and costmaps[0].info.width==pixels.shape[1]
                    and costmaps[0].info.height==pixels.shape[0]
                    and math.isclose(costmaps[0].info.resolution,info['resolution'],rel_tol=1e-6)
                    and any(value>=0 for value in costmaps[0].data)):
                break
        else:
            raise RuntimeError('Initialized global costmap unavailable; see saved node logs')
        def pose(values):
            p=PoseStamped();p.header.frame_id='map';p.header.stamp=node.get_clock().now().to_msg()
            p.pose.position.x=float(values[0]);p.pose.position.y=float(values[1])
            p.pose.orientation.z=math.sin(values[2]/2);p.pose.orientation.w=math.cos(values[2]/2)
            return p
        goal=ComputePathToPose.Goal();goal.start=pose(args.start);goal.goal=pose(args.goal);goal.use_start=True;goal.planner_id='NavFn'
        future=client.send_goal_async(goal);rclpy.spin_until_future_complete(node,future,timeout_sec=30)
        if not future.done() or future.result() is None or not future.result().accepted:
            raise RuntimeError('NavFn rejected path request or timed out')
        handle=future.result();future=handle.get_result_async();rclpy.spin_until_future_complete(node,future,timeout_sec=60)
        if not future.done() or future.result().status!=GoalStatus.STATUS_SUCCEEDED:
            raise RuntimeError('NavFn path computation failed or timed out')
        raw=np.array([[p.pose.position.x,p.pose.position.y] for p in future.result().result.path.poses])
        path=tangent_path(smooth_positions(raw,config['smoothing']))
        assert_clear(path,info,blocked,config['planner']['robot_radius_m'])
        np.savetxt(output/'navfn_positions.csv',raw,delimiter=',',header='x,y',comments='')
        np.savetxt(output/'E2_environment.csv',path,delimiter=',',header='x,y,yaw',comments='')
        plot_path(output/'preview.png',path,raw,info,pixels)
        metadata={'frame':'map','planner':'NavFn','start':args.start,'requested_goal':args.goal,
                  'terminal_yaw_policy':'forward tangent, including endpoint; requested goal yaw is not retained',
                  'csv_sha256':sha(output/'E2_environment.csv'),'map_yaml_sha256':sha(args.map),
                  'map_image_sha256':sha(args.map.parent/info['image']),'planner_params_sha256':sha(paramfile),
                  'path_config_sha256':sha(output/'path_config.yaml'),'smoothing':config['smoothing'],
                  'footprint_check':'circular footprint against occupied, unknown and outside-map cells',
                  'costmap_ready_before_goal':True,
                  'physical_trials':0,'software_manifest_status':'Regenerate after commit'}
        (output/'path_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        print(output/'E2_environment.csv')
    except Exception as exc:
        (output/'failure.json').write_text(json.dumps({'error':str(exc),'physical_trials':0},indent=2)+'\n')
        raise
    finally:
        for process in processes:
            if process.poll() is None: process.send_signal(signal.SIGINT)
        for process in processes:
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired: process.kill();process.wait()
        for log in logs: log.close()
        node.destroy_node();rclpy.shutdown()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map',type=Path,required=True)
    parser.add_argument('--start',type=float,nargs=3,required=True,metavar=('X','Y','YAW'))
    parser.add_argument('--goal',type=float,nargs=3,required=True,metavar=('X','Y','YAW'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path)
    args=parser.parse_args()
    if args.config is None:
        local=Path(__file__).resolve().parents[1]/'params/dwvp_access_path.yaml'
        if not local.exists():
            from ament_index_python.packages import get_package_share_directory
            local=Path(get_package_share_directory('dwpp_test_simulation'))/'params/dwvp_access_path.yaml'
        args.config=local
    generate(args)


if __name__=='__main__':
    main()
