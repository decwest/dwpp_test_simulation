import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
SPEC=importlib.util.spec_from_file_location('path_tool',ROOT/'scripts/dwvp_access_path.py')
path_tool=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(path_tool)


def settings():
    return yaml.safe_load((ROOT/'params/dwvp_access_path.yaml').read_text())['smoothing']


@pytest.mark.parametrize('direction',[(1,0),(-1,0),(0,1),(0,-1)])
def test_forward_tangent_never_flips_180_degrees(direction):
    xy=np.linspace([0,0],direction,30)
    path=path_tool.tangent_path(path_tool.resample_positions(xy,settings()['sample_spacing_m']))
    expected=np.asarray(direction,dtype=float)
    np.testing.assert_allclose(np.c_[np.cos(path[:,2]),np.sin(path[:,2])]@expected,1,atol=1e-12)


def test_resampling_pins_endpoints_and_limits_spacing():
    raw=np.array([[0.,0.],[0.4,0.],[0.4,0.6]])
    result=path_tool.resample_positions(raw,.02)
    np.testing.assert_allclose(result[[0,-1]],raw[[0,-1]])
    assert np.linalg.norm(np.diff(result,axis=0),axis=1).max() <= .02000000001
    assert len(result)==51


def test_symmetric_window_on_circle_and_angle_wrap():
    theta=np.linspace(2.,4.,1001)
    xy=np.c_[np.cos(theta),np.sin(theta)]
    path=path_tool.tangent_path(xy,.1)
    expected=theta+np.pi/2
    error=np.arctan2(np.sin(path[:,2]-expected),np.cos(path[:,2]-expected))
    np.testing.assert_allclose(error[51:-51],0.,atol=1e-9)
    report,profile,_=path_tool.path_diagnostics(path,.22,.6)
    assert report['excess_count']==0
    np.testing.assert_allclose(profile[60:-60,5],1.,atol=1e-5)


def test_feasibility_reports_both_signs_and_locations(tmp_path):
    import sys
    sys.path.insert(0,str(ROOT/'scripts'))
    x=np.linspace(0,1,101)
    path=np.c_[x,np.zeros(len(x)),4*x]
    report,profile,_=path_tool.path_diagnostics(path)
    assert report['excess_count']==101
    assert report['excess_points'][50]['progress_m']==.5
    np.testing.assert_allclose(profile[:,6],.88,atol=1e-12)
    reverse=path.copy();reverse[:,2]*=-1
    other,_,_=path_tool.path_diagnostics(reverse)
    assert other['excess_count']==101
    file=tmp_path/'route.csv';np.savetxt(file,path,delimiter=',',header='x,y,yaw',comments='')
    path_tool.check_path(file,tmp_path/'check')
    assert (tmp_path/'check/path_check.png').stat().st_size>0
    assert (tmp_path/'check/profile.csv').exists()


def test_footprint_and_between_sample_collision_detection():
    info={'resolution':.1,'origin':[0,0,0]};blocked=np.zeros((30,30),dtype=bool)
    path=np.array([[.5,1,0],[2.5,1,0]])
    path_tool.assert_clear(path,info,blocked,.22)
    blocked[10,15]=True
    with pytest.raises(ValueError,match='intersects'):
        path_tool.assert_clear(path,info,blocked,.22)


def test_rotated_map_coordinates():
    info={'resolution':.1,'origin':[3,4,np.pi/2]};blocked=np.zeros((30,30),dtype=bool)
    path=np.array([[2.5,4.5,0],[2.5,6.5,0]])
    path_tool.assert_clear(path,info,blocked,.1)


def test_unknown_and_outside_map_rejected():
    info={'resolution':.1,'origin':[0,0,0]};blocked=np.zeros((30,30),dtype=bool)
    with pytest.raises(ValueError,match='intersects'):
        path_tool.assert_clear(np.array([[.05,.5,0],[1,.5,0]]),info,blocked,.22)


@pytest.mark.parametrize('spacing',[0,-1,float('nan')])
def test_invalid_resampling_settings(spacing):
    with pytest.raises(ValueError,match='positive'):
        path_tool.resample_positions([[0,0],[1,0]],spacing)


def test_trinary_map_threshold_and_negate(tmp_path):
    from PIL import Image
    image=tmp_path/'map.pgm';Image.fromarray(np.array([[0,127,255]],dtype=np.uint8)).save(image)
    file=tmp_path/'map.yaml';file.write_text(yaml.safe_dump({'image':'map.pgm','resolution':.1,'origin':[0,0,0],
        'negate':0,'free_thresh':.2,'occupied_thresh':.65}))
    _,_,blocked=path_tool.load_map(file)
    assert blocked.tolist()==[[True,True,False]]


def test_subcell_reversal_removal_keeps_endpoints_and_large_bends():
    tiny=np.array([[0.,0.],[.05,0.],[.04,.001],[.1,.01]])
    cleaned,removed=path_tool.remove_subcell_cusps(tiny,.05)
    assert len(removed)==1
    np.testing.assert_allclose(cleaned[[0,-1]],tiny[[0,-1]])
    large=tiny*100
    cleaned,removed=path_tool.remove_subcell_cusps(large,.05)
    assert not removed
    np.testing.assert_allclose(cleaned,large)
