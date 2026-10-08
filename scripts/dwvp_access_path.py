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


def remove_subcell_cusps(xy, resolution):
    """Remove NavFn's sub-cell reversals, which native Humble otherwise pins.

    A forward-only NavFn route has no intended reverse segments. Only an interior
    negative-dot-product vertex with BOTH adjacent edges <= one cell diagonal is
    removed. Endpoints and larger-scale bends are preserved. Log every deletion;
    the final circular-footprint collision check still applies.
    """
    points = [np.asarray(p,dtype=float) for p in xy]
    removed = []
    i = 1
    while i < len(points)-1:
        before, after = points[i]-points[i-1], points[i+1]-points[i]
        if np.dot(before,after)<0 and max(np.linalg.norm(before),np.linalg.norm(after))<=math.sqrt(2)*resolution*(1+1e-6):
            removed.append(points.pop(i).tolist())
            i=max(1,i-1)
        else:
            i+=1
    return np.asarray(points), removed


def resample_positions(xy, spacing):
    """Resample AFTER Nav2 smoothing, preserving endpoints and at most spacing."""
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2 or not np.isfinite(xy).all():
        raise ValueError('At least two finite path positions are required')
    xy = xy[np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-9]]
    if len(xy) < 2:
        raise ValueError('Path has no length')
    if spacing <= 0 or not math.isfinite(spacing):
        raise ValueError('sample_spacing_m must be positive')
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    sample = np.linspace(0, arc[-1], max(2, int(math.ceil(arc[-1]/spacing))+1))
    return np.c_[np.interp(sample, arc, xy[:, 0]), np.interp(sample, arc, xy[:, 1])]


def tangent_path(xy, half_window_m=0.10):
    """Forward secant from s-0.10 to s+0.10 m, clipped at the endpoints."""
    xy = np.asarray(xy, dtype=float)
    if len(xy) < 2 or not np.isfinite(xy).all() or not math.isfinite(half_window_m) or half_window_m <= 0:
        raise ValueError('Finite positions and positive tangent window required')
    spacing = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    if np.any(spacing <= 1e-9):
        raise ValueError('Duplicate positions after smoothing')
    arc = np.r_[0., np.cumsum(spacing)]
    lo, hi = np.maximum(0., arc-half_window_m), np.minimum(arc[-1], arc+half_window_m)
    tangent = np.c_[np.interp(hi, arc, xy[:, 0])-np.interp(lo, arc, xy[:, 0]),
                    np.interp(hi, arc, xy[:, 1])-np.interp(lo, arc, xy[:, 1])]
    if np.any(np.linalg.norm(tangent, axis=1) <= 1e-9):
        raise ValueError('Undefined tangent (cusp) after smoothing')
    return np.c_[xy, np.arctan2(tangent[:, 1], tangent[:, 0])]


def path_diagnostics(path, speed=0.22, omega_max=0.6):
    from dwvp_access_experiment import validate_path
    path = validate_path(path)
    if not np.isfinite([speed, omega_max]).all() or speed <= 0 or omega_max <= 0:
        raise ValueError('Positive finite speed and yaw-rate limit required')
    spacing = np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)
    arc = np.r_[0., np.cumsum(spacing)]
    yaw = np.unwrap(path[:, 2])
    # Required orientation rate, not an estimate from three noisy position points.
    curvature = np.gradient(yaw, arc)
    required = speed*curvature
    step = np.diff(yaw)
    excess = np.flatnonzero(np.abs(required) > omega_max)
    # Also expose geometric segment curvature so yaw filtering cannot hide position wiggles.
    segment_yaw = np.unwrap(np.arctan2(*np.diff(path[:, :2], axis=0).T[::-1]))
    geometric = np.diff(segment_yaw)/((spacing[:-1]+spacing[1:])/2)
    report = dict(points=len(path), length_m=float(arc[-1]), speed_m_s=speed, omega_max_rad_s=omega_max,
        curvature_definition='gradient(unwrapped CSV yaw, cumulative XY chord length)',
        spacing_m=dict(min=float(spacing.min()), mean=float(spacing.mean()), max=float(spacing.max())),
        max_abs_curvature_per_m=float(np.abs(curvature).max()),
        geometric_max_abs_curvature_per_m=float(np.abs(geometric).max()) if len(geometric) else 0.,
        required_yaw_rate_abs_p90_rad_s=float(np.percentile(np.abs(required), 90)),
        max_abs_required_yaw_rate_rad_s=float(np.abs(required).max()),
        largest_orientation_step_deg=float(np.degrees(np.abs(step).max())),
        largest_step_end_progress_m=float(arc[np.argmax(np.abs(step))+1]),
        orientation_steps_over_5_deg=int(np.sum(np.abs(step)>np.deg2rad(5))),
        excess_count=len(excess), excess_share_pct=100.*len(excess)/len(path),
        excess_points=[dict(index=int(i), progress_m=float(arc[i]), x=float(path[i,0]), y=float(path[i,1]),
                            curvature_per_m=float(curvature[i]), required_yaw_rate_rad_s=float(required[i])) for i in excess])
    profile = np.c_[arc, path, np.degrees(np.r_[0., step]), curvature, required]
    return report, profile, geometric


def check_path(csv_file, output, speed=0.22, omega_max=0.6):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path = np.loadtxt(csv_file, delimiter=',', skiprows=1)
    report, profile, geometric = path_diagnostics(path, speed, omega_max)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    report.update(csv_sha256=sha(csv_file), source_csv=str(csv_file))
    np.savetxt(output/'profile.csv', profile, delimiter=',', comments='',
               header='progress_m,x,y,yaw,yaw_step_deg,curvature_per_m,required_yaw_rate_rad_s')
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    for ax, values, label in zip(axes, [np.degrees(np.unwrap(path[:,2])), profile[:,4], profile[:,5], profile[:,6]],
                               ['yaw [deg]', 'yaw step [deg]', 'curvature [1/m]', 'required yaw rate [rad/s]']):
        ax.plot(profile[:,0], values); ax.set_ylabel(label); ax.grid(True, alpha=.3)
    axes[2].plot(profile[1:-1,0], geometric, alpha=.4, label='XY segment curvature')
    axes[2].legend()
    for limit in (-omega_max, omega_max):
        axes[3].axhline(limit, color='red', linestyle='--')
    axes[-1].set_xlabel('progress [m]')
    fig.tight_layout(); fig.savefig(output/'path_check.png', dpi=160); plt.close(fig)
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


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
    from nav2_msgs.action import ComputePathToPose, SmoothPath
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
    if config['smoothing']['method'] != 'nav2_simple_smoother':
        raise ValueError('E2 requires Nav2 SimpleSmoother')
    params=planner_parameters(output/'map.yaml',config,info['resolution'])
    params['smoother_server'] = {'ros__parameters': {
        'smoother_plugins': ['simple_smoother'],
        'simple_smoother': {'plugin': 'nav2_smoother::SimpleSmoother',
            **{k: config['smoothing'][k] for k in ('w_data', 'w_smooth', 'max_its', 'tolerance', 'do_refinement')}}}}
    # Newer Nav2 can expose this parameter; this Humble build hardcodes four.
    if 'refinement_num' in config['smoothing']:
        params['smoother_server']['ros__parameters']['simple_smoother']['refinement_num'] = config['smoothing']['refinement_num']
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
        for package,executable in (('nav2_map_server','map_server'),('nav2_planner','planner_server'),('nav2_smoother','smoother_server')):
            spawn(package,executable,['--ros-args','--params-file',str(paramfile)])
        spawn('nav2_lifecycle_manager','lifecycle_manager',['--ros-args','-p','autostart:=true','-p',"node_names:=['map_server','planner_server','smoother_server']",'-p','bond_timeout:=0.0'])
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
        navfn_path = future.result().result.path
        raw=np.array([[p.pose.position.x,p.pose.position.y] for p in navfn_path.poses])
        removed = []
        if config['planner'].get('remove_subcell_cusps', False):
            cleaned, removed = remove_subcell_cusps(raw, info['resolution'])
            # Use only retained native poses; do not perform custom position smoothing.
            from dwvp_access_experiment import pose_path_message
            navfn_path = pose_path_message(tangent_path(cleaned), 'map', node.get_clock().now().to_msg())
        np.savetxt(output/'smoother_input_positions.csv',
                   [[p.pose.position.x,p.pose.position.y] for p in navfn_path.poses],
                   delimiter=',',header='x,y',comments='')
        smoother = ActionClient(node, SmoothPath, '/smooth_path')
        if not smoother.wait_for_server(timeout_sec=30):
            raise RuntimeError('Smoother action unavailable')
        # Action discovery precedes lifecycle activation; wait for ACTIVE explicitly.
        from lifecycle_msgs.srv import GetState
        from lifecycle_msgs.msg import State
        state_client = node.create_client(GetState, '/smoother_server/get_state')
        deadline = time.monotonic()+30
        while time.monotonic()<deadline:
            if state_client.wait_for_service(timeout_sec=.1):
                state_future = state_client.call_async(GetState.Request())
                rclpy.spin_until_future_complete(node,state_future,timeout_sec=1.)
                if state_future.done() and state_future.result().current_state.id==State.PRIMARY_STATE_ACTIVE:
                    break
            rclpy.spin_once(node,timeout_sec=.1)
        else:
            raise RuntimeError('Smoother lifecycle did not activate')
        node.destroy_client(state_client)
        from dwvp_access_experiment import snapshot_parameters
        runtime = snapshot_parameters(node, 'smoother_server')
        (output/'smoother_server_runtime.yaml').write_text(yaml.safe_dump(runtime))
        actual = runtime['/smoother_server']['ros__parameters']
        if ('refinement_num' in config['smoothing'] and 'simple_smoother.refinement_num' not in actual):
            raise ValueError('This Nav2 does not support refinement_num; omit it to use the documented native behavior')
        request = SmoothPath.Goal(); request.path = navfn_path
        request.smoother_id = 'simple_smoother'; request.max_smoothing_duration.sec = 10
        request.check_for_collisions = True
        future = smoother.send_goal_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=30)
        if not future.done() or future.result() is None or not future.result().accepted:
            raise RuntimeError('Smoother rejected request or timed out')
        future = future.result().get_result_async()
        rclpy.spin_until_future_complete(node, future, timeout_sec=30)
        if not future.done() or future.result().status != GoalStatus.STATUS_SUCCEEDED or not future.result().result.was_completed:
            raise RuntimeError('Nav2 smoothing failed or was incomplete')
        smoothed = np.array([[p.pose.position.x,p.pose.position.y] for p in future.result().result.path.poses])
        np.savetxt(output/'simple_smoother_positions.csv',smoothed,delimiter=',',header='x,y',comments='')
        path=tangent_path(resample_positions(smoothed,config['smoothing']['sample_spacing_m']),
                          config['smoothing']['tangent_half_window_m'])
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
                  'smoother':'nav2_smoother::SimpleSmoother', 'smoother_action_completed':True,
                  'removed_subcell_cusps':removed,
                  'refinement_num':actual.get('simple_smoother.refinement_num', 4),
                  'refinement_note':'Humble 1.1.19 uses four hardcoded refinements; requested default two is unsupported',
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
    import sys
    if len(sys.argv)>1 and sys.argv[1]=='check-path':
        parser=argparse.ArgumentParser(description='Diagnose a frozen CSV without changing it')
        parser.add_argument('command'); parser.add_argument('--path',type=Path,required=True)
        parser.add_argument('--output',type=Path,required=True)
        parser.add_argument('--speed',type=float,default=.22)
        parser.add_argument('--omega-max',type=float,default=.6)
        args=parser.parse_args()
        print(json.dumps(check_path(args.path,args.output,args.speed,args.omega_max),indent=2))
        return
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
