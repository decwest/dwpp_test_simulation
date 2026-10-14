"""Selection is based on completed common-metric integrals, never travel time."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from dwvp_access_e2 import grid, select


def test_v4_grid_and_e2_only_local_costmap_resolution():
    import yaml
    from dwvp_access_experiment import render_parameters
    root = Path(__file__).resolve().parents[1]
    old = yaml.safe_load((root/'params/dwvp_access_e2_route_design_v3.yaml').read_text())
    new = yaml.safe_load((root/'params/dwvp_access_e2_route_design_v4.yaml').read_text())
    params = root/'params/hsrb_dwvp_access_params.yaml'
    for name in old['conditions']:
        if name.startswith('E1'):
            assert old['conditions'][name] == new['conditions'][name]
            assert render_parameters(params, old, name) == render_parameters(params, new, name)
    a, b = (render_parameters(params, config, 'E2_environment') for config in (old, new))
    assert a['global_costmap'] == b['global_costmap']
    expected = a['local_costmap']['local_costmap']['ros__parameters'].copy()
    expected['resolution'] = .01
    assert expected == b['local_costmap']['local_costmap']['ros__parameters']
    candidates = grid('route-v4')
    assert len(candidates) == 8 and all(c == 'DWB' for c, _, _ in candidates)
    assert {(o['sim_time'], o['PathOrientationCritic.scale'], o['PathPositionCritic.scale']) for _, _, o in candidates} == {
        (t, y, p) for t in (2., 3.) for y in (4., 16.) for p in (16., 64.)}


def row(name,p,h,success=True):
    return dict(controller='MPPI',candidate=name,overrides={},success=success,
                eval_position_error_integral_m_s=p,eval_heading_error_integral_deg_s=h)


def test_selection_normalizes_each_integral_and_excludes_failures():
    rows=[row('position',1,10),row('balanced',2,2),row('orientation',10,1),row('failed',.01,.01,False)]
    selected=select(rows)
    assert selected['MPPI']['candidate']=='balanced'
    assert selected['MPPI']['score']==4
    assert selected['DWB'] is None


def test_zero_minimum_and_tie_are_defined():
    rows=[row('first',0,2),row('second',0,2),row('positive',1,1)]
    assert select(rows)['MPPI']['candidate']=='first'
    assert rows[-1]['selection_score'] is None


def test_grid_is_exact():
    candidates=grid()
    assert len(candidates)==9
    assert len([c for c in candidates if c[0]=='DWB'])==6
    assert [o['PathAngleCritic']['cost_weight'] for c,_,o in candidates if c=='MPPI']==[6.,12.,20.]


def test_route_design_config_preserves_e1_and_sets_requested_e2():
    import yaml
    root = Path(__file__).resolve().parents[1]
    original = yaml.safe_load((root/'params/dwvp_access_experiment.yaml').read_text())
    design = yaml.safe_load((root/'params/dwvp_access_e2_route_design.yaml').read_text())
    for name, condition in original['conditions'].items():
        if name.startswith('E1'):
            assert condition == design['conditions'][name]
    assert original['common'] == design['common']
    e2 = design['conditions']['E2_environment']
    assert e2['methods'] == ['VP_CLIP', 'DWVP', 'MPPI', 'DWB']
    assert e2['controller_overrides']['DWVP']['lateral_time_constant'] == .6
    for name in ('VP_CLIP', 'DWVP'):
        assert not e2['controller_overrides'][name]['use_cost_regulated_linear_velocity_scaling']
        assert not e2['controller_overrides'][name]['use_rotate_to_heading']


def test_route_v2_full_factorial_and_e1_preservation():
    import yaml
    candidates = grid('route-v2')
    assert len(candidates) == 48
    dwb = [o for c, _, o in candidates if c == 'DWB']
    mppi = [o for c, _, o in candidates if c == 'MPPI']
    assert len(dwb) == 36 and len(mppi) == 12
    assert len({(o['sim_time'], o['PathOrientationCritic.scale'], o['PathPositionCritic.scale']) for o in dwb}) == 36
    assert {o['sim_time'] for o in dwb} == {1.7, 2., 3., 8.}
    assert len({(o['PathAngleCritic']['cost_weight'], o['PathAlignCritic']['cost_weight'], o['wz_std']) for o in mppi}) == 12
    root = Path(__file__).resolve().parents[1]
    original = yaml.safe_load((root/'params/dwvp_access_experiment.yaml').read_text())
    design = yaml.safe_load((root/'params/dwvp_access_e2_route_design_v2.yaml').read_text())
    for name, condition in original['conditions'].items():
        if name.startswith('E1'):
            assert condition == design['conditions'][name]
    assert original['common'] == design['common']
    e2 = design['conditions']['E2_environment']
    assert e2['start_pose'] is None  # prepare uses the actual first CSV pose.
    o = e2['controller_overrides']
    assert o['DWVP']['lateral_time_constant'] == .6
    assert not o['DWVP']['use_velocity_scaled_lookahead_dist']
    assert o['DWVP']['lookahead_dist'] == .11
    assert o['DWB']['critics'] == ['Oscillation', 'BaseObstacle', 'ObstacleFootprint', 'GoalDist',
                                  'GoalAlign', 'RotateToGoal', 'PathOrientationCritic', 'PathPositionCritic']
    assert o['DWB']['trajectory_generator_name'] == 'dwb_plugins::LimitedAccelGenerator'
    assert o['DWB']['sim_period'] == 1/30
    assert o['MPPI']['time_steps']*o['MPPI']['model_dt'] == 2.


def test_route_v2_dwb_critic_replacement_removes_only_inactive_defaults():
    import yaml
    from dwvp_access_experiment import render_parameters
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root/'params/dwvp_access_e2_route_design_v2.yaml').read_text())
    params = root/'params/hsrb_dwvp_access_params.yaml'
    dwb = render_parameters(params, config, 'E2_environment')['controller_server']['ros__parameters']['DWB']
    assert not any(k.startswith(('PathDist.', 'PathAlign.')) for k in dwb)
    assert dwb['GoalAlign.scale'] == 24.
    assert dwb['PathPositionCritic.class'] == 'dwb_path_tracking_critics::PathPositionCritic'
    original = yaml.safe_load((root/'params/dwvp_access_experiment.yaml').read_text())
    for condition in original['conditions']:
        if condition.startswith('E1'):
            assert render_parameters(params, config, condition) == render_parameters(params, original, condition)


def test_route_v3_grid_and_e1_preservation():
    import yaml
    from dwvp_access_experiment import render_parameters
    root = Path(__file__).resolve().parents[1]
    original = yaml.safe_load((root/'params/dwvp_access_experiment.yaml').read_text())
    config = yaml.safe_load((root/'params/dwvp_access_e2_route_design_v3.yaml').read_text())
    params = root/'params/hsrb_dwvp_access_params.yaml'
    for condition in original['conditions']:
        if condition.startswith('E1'):
            assert render_parameters(params, original, condition) == render_parameters(params, config, condition)
    values = render_parameters(params, config, 'E2_environment')
    assert values['local_costmap']['local_costmap']['ros__parameters']['inflation_layer']['inflation_radius'] == .3
    assert values['global_costmap']['global_costmap']['ros__parameters']['inflation_layer']['inflation_radius'] == .3
    dwb = values['controller_server']['ros__parameters']['DWB']
    assert [dwb[k] for k in ('vx_samples', 'vy_samples', 'vtheta_samples')] == [20, 20, 20]
    entries = grid('route-v3')
    assert len(entries) == 24
    combinations = {(o['sim_time'], o['PathOrientationCritic.scale'], o['PathPositionCritic.scale']) for c, _, o in entries if c == 'DWB'}
    assert combinations == {(t, y, p) for t in (2., 3.) for y in (4., 16., 64.) for p in (16., 64.)}
    assert [e for e in entries if e[0] == 'MPPI'] == [e for e in grid('route-v2') if e[0] == 'MPPI']
