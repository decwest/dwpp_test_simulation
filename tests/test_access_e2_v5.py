"""The v5 sweep changes only the authorized E2 DWB parameters."""
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from dwvp_access_e2 import grid, merge
from dwvp_access_experiment import render_parameters


def test_v5_frozen_grid_and_rendered_contract():
    root = Path(__file__).resolve().parents[1]
    old = yaml.safe_load((root/'params/dwvp_access_e2_route_design_v4.yaml').read_text())
    new = yaml.safe_load((root/'params/dwvp_access_e2_route_design_v5.yaml').read_text())
    params = root/'params/hsrb_dwvp_access_params.yaml'
    assert old['common'] == new['common']
    for name in old['conditions']:
        if name.startswith('E1'):
            assert old['conditions'][name] == new['conditions'][name]
            assert render_parameters(params, old, name) == render_parameters(params, new, name)
    before, after = [render_parameters(params, c, 'E2_environment') for c in (old, new)]
    assert before['local_costmap'] == after['local_costmap']
    assert before['global_costmap'] == after['global_costmap']
    assert after['local_costmap']['local_costmap']['ros__parameters']['resolution'] == .01
    controllers = after['controller_server']['ros__parameters']
    original = before['controller_server']['ros__parameters']
    for method in ('VP_CLIP', 'DWVP', 'MPPI'):
        assert controllers[method] == original[method]
    expected = original['DWB'].copy()
    expected['GoalAlign.scale'] = 0.
    assert controllers['DWB'] == expected
    entries = grid('route-v5')
    assert len(entries) == 8 and all(c == 'DWB' for c, _, _ in entries)
    assert [(o['GoalDist.scale'], o['PathOrientationCritic.scale'], o['sim_time']) for _, _, o in entries] == [
        (g, y, t) for g in (24., 96.) for y in (16., 64.) for t in (2., 3.)]
    for _, _, override in entries:
        effective = controllers['DWB'].copy()
        merge(effective, override)
        assert effective['GoalAlign.scale'] == 0.
        assert 'GoalAlign' in effective['critics']
        assert effective['PathPositionCritic.scale'] == 64.
        assert [effective[k] for k in ('vx_samples', 'vy_samples', 'vtheta_samples')] == [7, 7, 7]
        assert effective['trajectory_generator_name'] == 'dwb_plugins::LimitedAccelGenerator'
        assert effective['sim_period'] == 1/30
        assert effective['max_speed_xy'] == .22
