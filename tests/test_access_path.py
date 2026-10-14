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


def test_v4_facing_transitions_are_separated_from_both_corners():
    path, graph = path_tool.rounded_graph([[2.5, .05], [-.05, .05], [-.05, -2.6], [2.5, -2.6]], .3, .02)
    faced, facing = path_tool.facing_profile(path, [-.05, -.8, -.05, -1.8, np.pi], .35)
    gaps = path_tool.facing_corner_gaps(faced, graph, facing)
    assert len(gaps) == 1 and gaps[0]['at_least_0p1_m']
    assert gaps[0]['entry_gap_m'] >= .2
    assert gaps[0]['exit_gap_m'] >= .15
    # Catch the old too-long facing interval overlapping the second arc.
    faced, facing = path_tool.facing_profile(path, [-.05, -.8, -.05, -2.05, np.pi], .35)
    gaps = path_tool.facing_corner_gaps(faced, graph, facing)
    assert gaps[0]['exit_gap_m'] < 0 and not gaps[0]['at_least_0p1_m']
    assert abs(np.arctan2(np.sin(path[0, 2]-np.pi), np.cos(path[0, 2]-np.pi))) < 1e-12


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


def test_via_concatenation_keeps_order_removes_only_shared_junctions():
    segments = [np.array([[2., 1.], [0., 1.]]),
                np.array([[0., 1.], [0., -2.]]),
                np.array([[0., -2.], [2., -2.]])]
    joined = path_tool.concatenate_segments(segments)
    np.testing.assert_allclose(joined, [[2., 1.], [0., 1.], [0., -2.], [2., -2.]])
    # NavFn tolerance can leave a small nonzero junction gap: do not drop it.
    segments[1][0, 0] = .01
    assert len(path_tool.concatenate_segments(segments)) == 5


def test_facing_shelf_profile_preserves_xy_and_tangents_outside_ramps():
    xy = np.c_[np.full(301, -.1), np.linspace(.4, -2.6, 301)]
    tangent = path_tool.tangent_path(xy)
    path, meta = path_tool.facing_profile(tangent, [-.1, -.8, -.1, -1.9, np.pi], .35)
    np.testing.assert_allclose(path[:, :2], xy)
    region = meta['intervals'][0]
    selected = slice(region['first_index'], region['last_index']+1)
    np.testing.assert_allclose(np.cos(path[selected, 2]), -1.)
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    outside = (arc < region['transition_start_m']) | (arc > region['transition_end_m'])
    np.testing.assert_allclose(path[outside, 2], tangent[outside, 2])
    # Mid-ramp smoothstep gives half of the shortest 90-degree turn.
    midpoint = np.argmin(abs(arc-(region['start_m']-.175)))
    assert abs(np.arctan2(np.sin(path[midpoint, 2]+3*np.pi/4), np.cos(path[midpoint, 2]+3*np.pi/4))) < .04
    assert np.max(abs(np.diff(np.unwrap(path[:, 2])))) < .07


def test_facing_interpolation_wraps_shortest_angle_and_limits_finite_segment():
    x = np.linspace(0, 3, 301)
    tangent = np.c_[x, np.zeros(len(x)), np.full(len(x), np.deg2rad(179.))]
    path, meta = path_tool.facing_profile(tangent, [1., 0., 2., 0., np.deg2rad(-179.)], .35)
    assert np.max(abs(np.diff(np.unwrap(path[:, 2])))) < .002
    assert meta['selected_points'] == 101
    assert len(meta['intervals']) == 1
    with pytest.raises(ValueError, match='No path points'):
        path_tool.facing_profile(tangent, [1., .61, 2., .61, 0.], .35)
    with pytest.raises(ValueError, match='nonzero length'):
        path_tool.facing_profile(tangent, [1., 0., 1., 0., 0.], .35)
    with pytest.raises(ValueError, match='outside the path'):
        path_tool.facing_profile(tangent, [0., 0., 2., 0., 0.], .35)


def test_clearance_is_cell_rectangle_distance_in_rotated_map():
    info = {'resolution': .1, 'origin': [3., 4., np.pi/2]}
    occupied = np.zeros((10, 10), dtype=bool)
    occupied[2, 5] = True
    # Local point (.4,.25) is .1 from the rectangle [.5,.6] x [.2,.3].
    distances, cells = path_tool.clearance_profile([[2.75, 4.4, 0]], info, occupied)
    np.testing.assert_allclose(distances, [.1], atol=1e-12)
    assert cells.tolist() == [[5, 2]]


def test_diagnostics_separates_orientation_rate_and_xy_radius(tmp_path):
    path = np.c_[np.linspace(0, 1, 101), np.zeros(101), np.linspace(0, 4, 101)]
    file = tmp_path/'path.csv'
    np.savetxt(file, path, delimiter=',', header='x,y,yaw', comments='')
    report = path_tool.check_path(file, tmp_path/'check')
    assert report['excess_count'] == 101
    assert report['tangent_yaw']['excess_count'] == 0
    assert report['xy_min_radius_m'] is None


def test_via_reversal_cleanup_deduplicates_new_adjacent_points():
    raw = np.array([[0., 0.], [.05, 0.], [.06, .01], [.05, 0.], [.05, -.1]])
    clean, removed = path_tool.remove_subcell_cusps(raw, .05, deduplicate=True)
    assert np.all(np.linalg.norm(np.diff(clean, axis=0), axis=1) > 1e-9)
    assert removed
    np.testing.assert_allclose(clean[[0, -1]], raw[[0, -1]])


def test_constrained_smoother_changes_only_requested_native_parameter():
    config = yaml.safe_load((ROOT/'params/dwvp_access_path_v2.yaml').read_text())
    name, plugin, params = path_tool.smoother_parameters(config)
    assert name == 'constrained_smoother'
    assert plugin == 'nav2_constrained_smoother/ConstrainedSmoother'
    assert params['ros__parameters'][name] == {'plugin': plugin, 'minimum_turning_radius': .3}
    legacy = yaml.safe_load((ROOT/'params/dwvp_access_path.yaml').read_text())
    assert config['planner'] == legacy['planner']
    name, plugin, params = path_tool.smoother_parameters(legacy)
    assert name == 'simple_smoother'
    assert params['ros__parameters'][name]['w_smooth'] == .3


def test_via_duplicate_removal_rechecks_new_neighbor_reversal():
    # Removing B in A,B,A creates a duplicate A; removing that duplicate must
    # expose and remove the remaining sub-cell reversing junction as well.
    xy = np.array([[0., .05], [.05, 0.], [.1, -.05], [.11, -.07],
                   [.1, -.05], [.125, -.047], [.15, -.045], [.2, -.043]])
    cleaned, removed = path_tool.remove_subcell_cusps(xy, .05, deduplicate=True)
    again, more = path_tool.remove_subcell_cusps(cleaned, .05, deduplicate=True)
    np.testing.assert_allclose(cleaned, again)
    assert not more and removed


def test_graph_right_angle_has_exact_radius_tangents_and_spacing():
    path, metadata = path_tool.rounded_graph([[0, 0], [1, 0], [1, 1]], .3, .02)
    arc = metadata['corners'][0]
    assert arc['radius_m'] == pytest.approx(.3)
    assert not metadata['radius_reductions']
    np.testing.assert_allclose(arc['entry'], [.7, 0], atol=1e-12)
    np.testing.assert_allclose(arc['exit'], [1, .3], atol=1e-12)
    curve = (path[:, 0] > .7+1e-9) & (path[:, 1] < .3-1e-9)
    np.testing.assert_allclose(np.linalg.norm(path[curve, :2]-[.7, .3], axis=1), .3, atol=1e-12)
    np.testing.assert_allclose(path[[0, -1]], [[0, 0, 0], [1, 1, np.pi/2]], atol=1e-12)
    spacing = np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)
    assert spacing.max() <= .02+1e-12
    assert spacing.min() > .019
    _, _, curvature = path_tool.path_diagnostics(path)
    assert 1/np.max(np.abs(curvature)) == pytest.approx(.3, abs=.0001)


def test_graph_shrinks_short_edge_fillets_and_rejects_reversal():
    path, metadata = path_tool.rounded_graph([[0, 0], [1, 0], [1, .1], [2, .1]])
    assert len(metadata['radius_reductions']) == 2
    assert all(c['radius_m'] == pytest.approx(.05) for c in metadata['corners'])
    assert all(p['end_m'] >= p['start_m'] for p in metadata['primitives'])
    assert np.isfinite(path).all()
    with pytest.raises(ValueError, match='reversing'):
        path_tool.rounded_graph([[0, 0], [1, 0], [0, 0]])
    with pytest.raises(ValueError, match='distinct'):
        path_tool.rounded_graph([[0, 0], [0, 0], [1, 0]])


def test_graph_facing_keeps_exact_xy_and_outside_tangents():
    path, _ = path_tool.rounded_graph([[2.5, .05], [-.15, .05], [-.15, -2.5], [2.5, -2.6]])
    faced, metadata = path_tool.facing_profile(path, [-.15, -.8, -.15, -1.9, np.pi], .35)
    np.testing.assert_array_equal(path[:, :2], faced[:, :2])
    r = metadata['intervals'][0]
    np.testing.assert_allclose(np.cos(faced[r['first_index']:r['last_index']+1, 2]), -1)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1))]
    outside = (arc < r['transition_start_m']) | (arc > r['transition_end_m'])
    np.testing.assert_allclose(path[outside, 2], faced[outside, 2])


def test_inflation_profile_counts_lethal_unknown_and_rotated_grid():
    grid = np.array([[0, 90, 99], [100, -1, 0]])
    path = np.array([[.5, .5], [1.5, .5], [2.5, .5], [.5, 1.5], [1.5, 1.5], [4, 4]])
    result = path_tool.inflation_profile(path, grid, [0, 0, 0], 1.)
    assert result['costs'] == [0, 90, 99, 100, -1, -2]
    assert not result['traversable']
    assert result['lethal_count'] == result['inscribed_count'] == 1
    rotated = np.c_[-path[:, 1]+2, path[:, 0]+3]
    assert path_tool.inflation_profile(rotated, grid, [2, 3, np.pi/2], 1.)['costs'] == result['costs']
