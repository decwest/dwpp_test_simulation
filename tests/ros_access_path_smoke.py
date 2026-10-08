"""NavFn action + smoothing + frozen E2 import in an isolated container."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import yaml

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert Path('/.dockerenv').exists() and set(os.listdir('/sys/class/net')) == {'lo'}
    assert os.environ.get('ROS_LOCALHOST_ONLY') == '1'
    args.output.mkdir(parents=True,exist_ok=False)
    pixels=np.full((80,80),255,dtype=np.uint8)
    pixels[[0,-1],:]=0;pixels[:,[0,-1]]=0
    pixels[32:48,35:45]=0  # central static obstacle, clear routes above and below
    Image.fromarray(pixels).save(args.output/'map.pgm')
    (args.output/'map.yaml').write_text(yaml.safe_dump({'image':'map.pgm','resolution':.05,
        'origin':[0.,0.,0.],'negate':0,'occupied_thresh':.65,'free_thresh':.2}))
    subprocess.run([sys.executable,str(ROOT/'scripts/dwvp_access_path.py'),'--map',str(args.output/'map.yaml'),
        '--start','.6','2.0','0','--goal','3.4','2.0','0','--output',str(args.output/'route')],check=True,timeout=160)
    # The offline generator must finish its nodes before another ROS check starts.
    lingering=[]
    for command_line in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            executable=command_line.read_bytes().split(b'\0')[0].decode()
        except (OSError, UnicodeError):
            continue
        if Path(executable).name in ('map_server','planner_server','smoother_server','lifecycle_manager','static_transform_publisher'):
            lingering.append(executable)
    assert not lingering, f'Path generator left node processes alive: {lingering}'
    route=args.output/'route/E2_environment.csv'
    data=np.loadtxt(route,delimiter=',',skiprows=1)
    assert len(data)>10 and np.isfinite(data).all()
    metadata=json.loads((route.parent/'path_metadata.json').read_text())
    assert metadata['planner']=='NavFn' and metadata['smoothing']['method']=='nav2_simple_smoother'
    assert metadata['costmap_ready_before_goal']
    raw=np.loadtxt(route.parent/'navfn_positions.csv',delimiter=',',skiprows=1)
    np.testing.assert_allclose(data[[0,-1],:2],raw[[0,-1]],atol=1e-9)
    # The obstacle must force a detour, and headings must follow the saved positions.
    assert np.max(np.abs(data[:,1]-2.0))>.4
    sys.path.insert(0,str(ROOT/'scripts'))
    from dwvp_access_path import tangent_path
    np.testing.assert_allclose(data,tangent_path(data[:,:2],.10),atol=1e-9)
    assert metadata['smoother_action_completed']
    assert (route.parent/'smoother_server_runtime.yaml').exists()
    assert (route.parent/'preview.png').stat().st_size>0
    spec=importlib.util.spec_from_file_location('experiment',ROOT/'scripts/dwvp_access_experiment.py')
    experiment=importlib.util.module_from_spec(spec);spec.loader.exec_module(experiment)
    session=args.output/'session'
    experiment.prepare(session,ROOT/'params/hsrb_dwvp_access_params.yaml',[0.,0.,0.],route)
    manifest,trial,path=experiment.load_trial(session,'E2_environment_DWVP_r1')
    np.testing.assert_allclose(path,data)
    np.testing.assert_allclose(experiment.start_pose(manifest,trial),[.6,2.,0.])
    (args.output/'report.json').write_text(json.dumps({'physical_trials':0,'path_samples':len(path),
        'planned_trials':len(manifest['trials']),'planner':'NavFn','smoothing':'nav2_simple_smoother',
        'tangent_yaw_verified':True,'node_shutdown_verified':True,'costmap_ready_before_goal':True,
        'status':'synthetic_pass'},indent=2)+'\n')


if __name__=='__main__':
    main()
