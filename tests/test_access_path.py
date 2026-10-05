import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('path_tool',ROOT/'scripts/dwvp_access_path.py')
path_tool=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(path_tool)


def settings():
    return yaml.safe_load((ROOT/'params/dwvp_access_path.yaml').read_text())['smoothing']


@pytest.mark.parametrize('direction',[(1,0),(-1,0),(0,1),(0,-1)])
def test_forward_tangent_never_flips_180_degrees(direction):
    xy=np.linspace([0,0],direction,30)
    path=path_tool.tangent_path(path_tool.smooth_positions(xy,settings()))
    expected=np.asarray(direction,dtype=float)
    np.testing.assert_allclose(np.c_[np.cos(path[:,2]),np.sin(path[:,2])]@expected,1,atol=1e-12)


def test_smoothing_pins_endpoints_and_reduces_grid_roughness():
    raw=np.c_[np.linspace(0,1,51),.02*np.tile([1,-1],26)[:51]]
    smooth=path_tool.smooth_positions(raw,settings())
    np.testing.assert_allclose(smooth[[0,-1]],raw[[0,-1]])
    assert np.abs(np.diff(smooth[:,1],n=2)).mean()<np.abs(np.diff(raw[:,1],n=2)).mean()


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


def test_invalid_smoothing_settings():
    config=settings();config['smooth_weight']=1
    with pytest.raises(ValueError,match='weights'):
        path_tool.smooth_positions([[0,0],[1,0]],config)


def test_trinary_map_threshold_and_negate(tmp_path):
    from PIL import Image
    image=tmp_path/'map.pgm';Image.fromarray(np.array([[0,127,255]],dtype=np.uint8)).save(image)
    file=tmp_path/'map.yaml';file.write_text(yaml.safe_dump({'image':'map.pgm','resolution':.1,'origin':[0,0,0],
        'negate':0,'free_thresh':.2,'occupied_thresh':.65}))
    _,_,blocked=path_tool.load_map(file)
    assert blocked.tolist()==[[True,True,False]]
