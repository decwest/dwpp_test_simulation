import importlib.util
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('experiment', ROOT / 'scripts/dwvp_access_experiment.py')
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def test_balanced_reproducible_65_trial_schedule(tmp_path):
    params = ROOT / 'params/hsrb_dwvp_access_params.yaml'
    one = experiment.prepare(tmp_path / 'one', params)
    two = experiment.prepare(tmp_path / 'two', params)
    assert len(one['trials']) == len({t['id'] for t in one['trials']}) == 65
    assert one['trials'] == two['trials']
    counts = Counter((t['task'], t['controller']) for t in one['trials'])
    assert all(n == 5 for n in counts.values())
    assert len(counts) == 13
    assert {k[1] for k in counts} == set(experiment.CONTROLLERS)
    for repeat in range(1, 6):
        for task in experiment.CONDITIONS:
            block = [t for t in one['trials'] if t['repeat'] == repeat and t['task'] == task]
            assert {t['controller'] for t in block} == set(experiment.default_config()['conditions'][task]['methods'])
    assert one['paths']['E2_environment']['file'] is None
    assert experiment.summarize(tmp_path / 'one')['pending'] == 65


def test_condition_geometry_and_configurable_start():
    lateral = experiment.canonical_path('E1_lateral')
    np.testing.assert_allclose(lateral[[0,-1]], [[0,0,0],[2.5,0,0]])
    ramp = experiment.canonical_path('E1_orientation_nominal')
    np.testing.assert_allclose(ramp[[0,100,115,130,-1]],
                               [[0,0,0],[1,0,0],[1.15,0,np.pi/4],[1.3,0,np.pi/2],[2.5,0,np.pi/2]])
    np.testing.assert_array_equal(ramp, experiment.canonical_path('E1_orientation_half'))
    assert len(ramp) == 251 and np.all(np.diff(ramp[:,2]) >= 0)
    config = experiment.default_config()
    config['conditions']['E1_lateral']['length_m'] = 1.8
    assert experiment.canonical_path('E1_lateral', config)[-1,0] == 1.8


def test_freezes_map_origin_and_rejects_modified_inputs(tmp_path):
    params = ROOT / 'params/hsrb_dwvp_access_params.yaml'
    experiment.prepare(tmp_path / 'one', params, [3., 4., np.pi / 2])
    _, _, path = experiment.load_trial(tmp_path / 'one', 'E1_orientation_nominal_DWVP_r1')
    np.testing.assert_allclose(path[0], [3., 4., np.pi / 2])
    np.testing.assert_allclose(path[-1, :2], [3., 6.5])
    with pytest.raises(ValueError, match='not been supplied'):
        experiment.load_trial(tmp_path / 'one', 'E2_environment_DWVP_r1')
    (tmp_path / 'one/nav2_params_E1_orientation_nominal.yaml').write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        experiment.load_trial(tmp_path / 'one', 'E1_orientation_nominal_DWVP_r1')


def test_unset_origin_never_runs(tmp_path):
    experiment.prepare(tmp_path / 'one', ROOT / 'params/hsrb_dwvp_access_params.yaml')
    with pytest.raises(ValueError, match='origin'):
        experiment.load_trial(tmp_path / 'one', 'E1_lateral_DWPP_r1')


def test_status_reports_unready_and_incomplete_trials_without_writing(tmp_path):
    root = tmp_path / 'session'
    experiment.prepare(root, ROOT / 'params/hsrb_dwvp_access_params.yaml', [0, 0, 0])
    before = set(root.rglob('*'))
    states = {r['trial']: r['status'] for r in experiment.session_status(root)}
    assert len(states) == 65 and states['E2_environment_RPP_r1'] == 'needs_route'
    assert states['E1_orientation_nominal_DWVP_r1'] == 'pending'
    assert set(root.rglob('*')) == before
    folder = root / 'runs/E1_orientation_nominal_DWVP_r1'
    folder.mkdir(parents=True)
    states = {r['trial']: r['status'] for r in experiment.session_status(root)}
    assert states['E1_orientation_nominal_DWVP_r1'] == 'incomplete'


@pytest.mark.parametrize('source,received,frame,ranges,valid', [
    (10, 10, 'laser', [float('inf')] * 5, True),
    (10, 10, 'laser', [.8, float('nan')], True),
    (8, 10, 'laser', [.8], False),
    (10, 8, 'laser', [.8], False),
    (11, 10, 'laser', [.8], False),
    (10, 10, '', [.8], False),
    (10, 10, 'laser', [], False),
    (10, 10, 'laser', [float('nan')], False),
])
def test_laser_preflight_checks_source_time_and_usable_measurements(source, received, frame, ranges, valid):
    scan = SimpleNamespace(header=SimpleNamespace(frame_id=frame, stamp=SimpleNamespace(sec=source, nanosec=0)),
                           ranges=ranges, range_min=.05, range_max=5.)
    if valid:
        assert experiment.scan_quality(scan, received, 10.1)['frame'] == 'laser'
    else:
        with pytest.raises(RuntimeError):
            experiment.scan_quality(scan, received, 10.1)


def test_errors_use_same_projected_position_and_wrapped_yaw():
    path = np.array([[0., 0., np.deg2rad(170)], [2., 0., np.deg2rad(-170)]])
    errors = experiment.tracking_errors(np.array([[1., .2, np.pi]]), path)
    np.testing.assert_allclose(errors, [[.2, 0]], atol=1e-12)


def test_invalid_reference_rejected():
    with pytest.raises(ValueError, match='Duplicate'):
        experiment.validate_path([[0, 0, 0], [0, 0, 1]])
    with pytest.raises(ValueError, match='finite'):
        experiment.validate_path([[0, 0, 0], [1, 0, np.nan]])


def test_failed_runs_remain_in_counts_and_never_get_success_means(tmp_path):
    root = tmp_path / 'session'
    experiment.prepare(root, ROOT / 'params/hsrb_dwvp_access_params.yaml', [0, 0, 0])
    folder = root / 'runs/E1_lateral_DWPP_r1'
    folder.mkdir(parents=True)
    (folder / 'result.json').write_text(json.dumps({'status': 'timeout', 'success': False, 'duration_s': 120}))
    (folder / 'tracking.csv').write_text('t,stamp_s,x,y,yaw,tf_age_s,raw_age_s,applied_age_s,odom_age_s,odom_source_age_s\n0,1,0,0,0,0,0,0,0,0\n')
    np.savetxt(folder / 'reference.csv', experiment.canonical_path('E1_lateral'), delimiter=',', header='x,y,yaw', comments='')
    report = experiment.summarize(root)
    assert report['recorded'] == 1 and report['pending'] == 64
    assert next(g for g in report['groups'] if g['task'] == report['trials'][0]['task'] and g['controller'] == report['trials'][0]['controller'])['succeeded'] == 0
    assert next(g for g in report['groups'] if g['task'] == report['trials'][0]['task'] and g['controller'] == report['trials'][0]['controller'])['travel_time_s_mean'] is None


def test_command_diagnostics_use_prior_applied_not_raw(tmp_path):
    import sys
    sys.path.insert(0, str(ROOT/'scripts'))
    from dwvp_access_metrics import command_metrics
    header='stamp_s,source_stamp_s,vx,vy,omega\n'
    (tmp_path/'raw.csv').write_text(header+'1,,0.1,0,0\n1.04,,0.1,0,0\n')
    (tmp_path/'applied.csv').write_text(header+'0.99,,0,0,0\n1.03,,0.1,0,0\n1.06,,0.1,0,0\n')
    result = command_metrics(tmp_path, {'start_stamp_s':1,'duration_s':.1}, experiment.default_config()['common'])
    assert result['vx_increment_excess_ratio_max'] == pytest.approx(.1*30/.22-1)
    assert result['vx_increment_excess_time_s'] == pytest.approx(1/30)
    assert result['prior_applied_missing_or_stale'] == 0
    assert result['vx_velocity_excess_time_s'] == 0


def test_runtime_configuration_mismatch_is_detected(tmp_path):
    expected = ROOT / 'params/hsrb_dwvp_access_params.yaml'
    config = experiment.render_parameters(expected, experiment.default_config())
    expected = tmp_path / 'expected.yaml'
    expected.write_text(yaml.safe_dump(config))
    runtime_files = {}
    for name in ('controller_server', 'velocity_smoother'):
        file = tmp_path / (name + '.yaml')
        file.write_text(yaml.safe_dump({'/' + name: config[name]}))
        runtime_files[name] = file
    experiment.verify_runtime_parameters(expected, runtime_files, 'DWVP')
    config['velocity_smoother']['ros__parameters']['max_accel'] = [99., 99., 99.]
    runtime_files['velocity_smoother'].write_text(yaml.safe_dump({'/velocity_smoother': config['velocity_smoother']}))
    with pytest.raises(ValueError, match='Runtime parameter mismatch'):
        experiment.verify_runtime_parameters(expected, runtime_files, 'DWVP')


@pytest.mark.parametrize('controller,field,value', [
    ('DWVP', 'lookahead_time', 99.),
    ('DWVP', 'min_vel_y', -.9),
    ('DWVP', 'approach_velocity_scaling_dist', 0.),
    ('DWVP', 'min_orientation_time', 8.),
    ('RPP', 'desired_linear_vel', 9.),
    ('DWPP', 'max_linear_accel', 9.),
    ('MPPI', 'PathAlignCritic', {'enabled': False}),
])
def test_selected_controller_tuning_is_frozen(tmp_path, controller, field, value):
    expected = ROOT / 'params/hsrb_dwvp_access_params.yaml'
    config = experiment.render_parameters(expected, experiment.default_config())
    expected = tmp_path / 'expected.yaml'
    expected.write_text(yaml.safe_dump(config))
    config['controller_server']['ros__parameters'][controller][field] = value
    files = {}
    for name in ('controller_server', 'velocity_smoother'):
        files[name] = tmp_path / (name + '.yaml')
        files[name].write_text(yaml.safe_dump({'/' + name: config[name]}))
    with pytest.raises(ValueError, match='Runtime parameter mismatch'):
        experiment.verify_runtime_parameters(expected, files, controller)


@pytest.mark.parametrize('missing_prefix,source_age,expected_valid', [
    (.02, 0., 1), (.4, 0., 0), (.02, 10., 0), (.02, -.5, 0),
])
def test_summary_retains_prefix_and_rejects_stale_source(tmp_path, missing_prefix, source_age, expected_valid):
    root = tmp_path / 'session'
    experiment.prepare(root, ROOT / 'params/hsrb_dwvp_access_params.yaml', [0, 0, 0])
    folder = root / 'runs/E1_orientation_nominal_DWVP_r1'
    folder.mkdir(parents=True)
    (folder / 'result.json').write_text(json.dumps({'status': 'succeeded', 'success': True, 'duration_s': 1.,
                                                  'missing_command_prefix_s': missing_prefix}))
    (folder / 'tracking.csv').write_text(
        't,stamp_s,x,y,yaw,tf_age_s,raw_age_s,applied_age_s,odom_age_s,odom_source_age_s\n'
        f'.005,1,0,0,0,0,inf,inf,0,{source_age}\n'
        f'.5,1.5,0,0,0,0,0,0,0,{source_age}\n')
    np.savetxt(folder / 'reference.csv', experiment.canonical_path('E1_orientation_nominal'), delimiter=',', header='x,y,yaw', comments='')
    report = experiment.summarize(root)
    assert report['trials'][0]['samples'] == 2
    assert report['trials'][0]['missing_command_prefix_s'] == missing_prefix
    assert report['trials'][0]['stale_or_missing_samples'] >= 1
    assert report['trials'][0]['fresh_pose_samples'] == (2 if source_age == 0. else 0)
    if source_age != 0.:
        assert report['trials'][0]['position_rmse_m'] is None
    assert next(g for g in report['groups'] if g['task'] == report['trials'][0]['task'] and g['controller'] == report['trials'][0]['controller'])['valid_successes'] == expected_valid


def test_start_is_offset_and_transformed_not_reference_origin(tmp_path):
    root=tmp_path/'session'
    experiment.prepare(root, ROOT/'params/hsrb_dwvp_access_params.yaml', [3,4,np.pi/2])
    manifest,trial,path=experiment.load_trial(root,'E1_lateral_DWVP_r1')
    np.testing.assert_allclose(experiment.start_pose(manifest,trial),[2.5,4,np.pi/2])
    experiment.check_start_pose([2.5,4,np.pi/2],manifest,trial)
    with pytest.raises(RuntimeError,match='condition start'):
        experiment.check_start_pose(path[0],manifest,trial)
    with pytest.raises(FileExistsError):
        experiment.prepare(root,ROOT/'params/hsrb_dwvp_access_params.yaml',[0,0,0])


def test_environment_path_already_uses_map_coordinates(tmp_path):
    csv=tmp_path/'E2_environment.csv'
    csv.write_text('x,y,yaw\n3,4,1.5\n3,5,1.5\n')
    root=tmp_path/'session'
    experiment.prepare(root,ROOT/'params/hsrb_dwvp_access_params.yaml',[20,30,.2],csv)
    manifest,trial,path=experiment.load_trial(root,'E2_environment_DWVP_r1')
    np.testing.assert_allclose(path,[[3,4,1.5],[3,5,1.5]])
    np.testing.assert_allclose(experiment.start_pose(manifest,trial),path[0])


def test_nominal_tuning_is_rendered_for_all_methods(tmp_path):
    config=experiment.default_config();config['common']['lookahead_time']=.9
    value=experiment.render_parameters(ROOT/'params/hsrb_dwvp_access_params.yaml',config)
    cs=value['controller_server']['ros__parameters']
    for name in experiment.CONTROLLERS:
        assert cs[name]['plugin']=='dwpp_test_simulation::TimedController'
        assert cs[name]['wrapped_plugin']!='dwpp_test_simulation::TimedController'
        if name not in ('MPPI','DWB'):
            assert cs[name]['lookahead_time']==.9
            assert cs[name]['min_lookahead_dist']==.11 and cs[name]['max_lookahead_dist']==.33
    assert cs['MPPI']['PathAlignCritic']['use_path_orientations'] is False
    assert cs['DWVP']['use_dynamic_window_vector_pursuit'] is True
    assert cs['VP_CLIP']['use_dynamic_window_vector_pursuit'] is False
    assert {k:v for k,v in cs['VP_CLIP'].items() if k!='use_dynamic_window_vector_pursuit'} == {
        k:v for k,v in cs['DWVP'].items() if k!='use_dynamic_window_vector_pursuit'}


def test_config_changes_rejected(tmp_path):
    root=tmp_path/'s';experiment.prepare(root,ROOT/'params/hsrb_dwvp_access_params.yaml',[0,0,0])
    with (root/'experiment_config.yaml').open('a') as f:f.write('\n# changed\n')
    with pytest.raises(ValueError,match='changed'):
        experiment.load_trial(root,'E1_lateral_DWVP_r1')


def make_attempt(tmp_path,condition='E1_lateral',controller='DWVP',poses=None):
    root=tmp_path/'s'
    csv=tmp_path/'e2.csv';csv.write_text('x,y,yaw\n0,0,0\n2,0,0\n')
    experiment.prepare(root,ROOT/'params/hsrb_dwvp_access_params.yaml',[0,0,0],csv)
    name=f'{condition}_{controller}_r1';folder=root/'runs'/name;folder.mkdir(parents=True)
    _,_,path=experiment.load_trial(root,name)
    np.savetxt(folder/'reference.csv',path,delimiter=',',header='x,y,yaw',comments='')
    experiment.write_json(folder/'result.json',{'status':'succeeded','success':True,'start_stamp_s':10.,
        'duration_s':1.,'missing_command_prefix_s':.01,'final_position_error_m':.02,'final_yaw_error_rad':.03})
    poses=poses if poses is not None else [[0,.5,0],[.1,.04,0],[.2,-.015,0]]
    with (folder/'tracking.csv').open('w') as f:
        f.write('t,stamp_s,x,y,yaw,tf_age_s,raw_age_s,applied_age_s,odom_age_s,odom_source_age_s,scan_age_s,scan_source_age_s,scan_min_range_m,speed_m_s\n')
        for t,pose in zip(np.linspace(0,1,len(poses)),poses):
            f.write(','.join(map(str,[t,10+t,*pose,0,0,0,0,0,0,0,.4,.2]))+'\n')
    return root,folder,name


def test_lateral_crossing_deadband_and_convergence_distance(tmp_path):
    root,_,_=make_attempt(tmp_path)
    row=experiment.summarize(root)['trials'][0]
    assert row['lateral_crossing_raw_m']==pytest.approx(.015)
    assert row['lateral_crossing_beyond_deadband_m']==pytest.approx(.005)
    assert row['lateral_convergence_time_s']==pytest.approx(.5)
    assert row['lateral_convergence_distance_m']==pytest.approx(np.hypot(.1,.46))


def test_deadband_rejects_localization_sized_crossing(tmp_path):
    root,_,_=make_attempt(tmp_path,poses=[[0,.5,0],[.1,-.009,0]])
    assert experiment.summarize(root)['trials'][0]['lateral_crossing_beyond_deadband_m']==0


def test_orientation_signed_errors_speed_and_prediction(tmp_path):
    root,_,_=make_attempt(tmp_path,'E1_orientation_nominal','VP_SCALED',
                          [[.5,0,.1],[1.15,0,np.pi/4-.2],[2.5,0,np.pi/2]])
    row=experiment.summarize(root)['trials'][0]
    assert row['yaw_lead_max_rad'] == pytest.approx(.1)
    assert row['yaw_lag_max_rad'] == pytest.approx(.2)
    assert row['orientation_min_speed_m_s'] == .2
    assert row['orientation_speed_samples'] == 1
    assert row['orientation_predicted_speed_m_s'] == pytest.approx(.6/(np.pi/2/.3))
    assert row['final_position_error_m']==.02 and row['final_yaw_error_rad']==.03
    assert row['yaw_error_role']=='tracking'


def test_environment_scan_speed_without_invented_survey_clearance(tmp_path):
    root,_,_=make_attempt(tmp_path,'E2_environment',poses=[[0,0,0],[.1,0,0]])
    row=experiment.summarize(root)['trials'][0]
    assert row['obstacle_near_mean_speed_m_s']==.2 and row['obstacle_near_samples']==2
    assert row['surveyed_min_clearance_m'] is None


def test_incomplete_and_missing_files_remain_counted(tmp_path):
    root=tmp_path/'s';experiment.prepare(root,ROOT/'params/hsrb_dwvp_access_params.yaml',[0,0,0])
    (root/'runs/E1_lateral_DWPP_r1').mkdir(parents=True)
    report=experiment.summarize(root)
    assert report['recorded']==1 and report['pending']==64 and len(report['groups'])==13
    assert report['trials'][0]['status']=='incomplete'
    assert 'tracking.csv' in report['trials'][0]['data_errors']
    group=next(g for g in report['groups'] if g['task']=='E1_lateral' and g['controller']=='DWPP')
    assert group['failed_or_incomplete']==1 and group['success_rate_attempted']==0


def test_timing_gaps_exceptions_and_missing_samples(tmp_path):
    root,folder,name=make_attempt(tmp_path)
    (folder/'timing.csv').write_text('receive_stamp_s,stamp_s,controller,sequence,compute_time_ms,success\n'
        '10.1,10.1,DWVP,1,2,true\n10.2,10.2,DWVP,3,4,false\n10.3,10.3,RPP,4,100,true\n')
    report=experiment.summarize(root)
    timing=report['controller_timing'][name]
    assert timing['samples']==2 and timing['sequence_gaps']==1 and timing['failed_calls']==1
    assert timing['mean_ms']==3 and timing['max_ms']==4


def test_survey_geometry_distances():
    import sys
    sys.path.insert(0,str(ROOT/'scripts'))
    from dwvp_access_metrics import obstacle_distance
    xy=np.array([[0,0],[2,0]])
    np.testing.assert_allclose(obstacle_distance(xy,[{'type':'circle','center':[0,0],'radius':.5}]),[-.5,1.5])
    np.testing.assert_allclose(obstacle_distance(xy,[{'type':'polygon','vertices':[[-1,-1],[1,-1],[1,1],[-1,1]]}]),[-1,1])


def test_constraint_metrics_use_configured_limits_and_count_unpaired(tmp_path):
    import sys
    sys.path.insert(0,str(ROOT/'scripts'))
    from dwvp_access_metrics import command_metrics
    (tmp_path/'raw.csv').write_text('stamp_s,vx,vy,omega\n1,.6,0,0\n1.1,.1,0,0\n')
    config=experiment.default_config()['common'];config['max_velocity']=[.3,.3,.6];config['min_velocity']=[-.3,-.3,-.6]
    result=command_metrics(tmp_path,{'start_stamp_s':1,'duration_s':.2},config)
    assert result['vx_velocity_excess_ratio_max']==1
    assert result['vx_velocity_excess_time_s']==pytest.approx(1/30)
    assert result['prior_applied_missing_or_stale']==2
    assert result['vx_increment_excess_ratio_max'] is None


def test_bad_laser_samples_are_missing_not_clear_space():
    scan=SimpleNamespace(ranges=[np.nan,-np.inf,.01],range_min=.05,range_max=5.)
    assert np.isnan(experiment.scan_minimum_range(scan))
    scan.ranges=[np.inf,np.nan]
    assert experiment.scan_minimum_range(scan)==np.inf
    scan.ranges=[np.inf,.3,.5]
    assert experiment.scan_minimum_range(scan)==.3


def test_seed_changes_method_order_without_changing_paths(tmp_path):
    params=ROOT/'params/hsrb_dwvp_access_params.yaml'
    one=experiment.prepare(tmp_path/'one',params,seed=1)
    two=experiment.prepare(tmp_path/'two',params,seed=2)
    assert one['trials']!=two['trials']
    assert one['paths']==two['paths']


@pytest.mark.parametrize('velocity',[-.4,.4])
def test_negative_and_positive_increment_constraints(tmp_path,velocity):
    import sys
    sys.path.insert(0,str(ROOT/'scripts'))
    from dwvp_access_metrics import command_metrics
    (tmp_path/'raw.csv').write_text(f'stamp_s,vx,vy,omega\n1,{velocity},0,0\n')
    (tmp_path/'applied.csv').write_text('stamp_s,vx,vy,omega\n.99,0,0,0\n')
    result=command_metrics(tmp_path,{'start_stamp_s':1,'duration_s':.1},experiment.default_config()['common'])
    assert result['vx_increment_excess_ratio_max']==pytest.approx(.4*30/.22-1)
    assert result['vx_velocity_excess_ratio_max']==pytest.approx(.4/.22-1)


def test_constraint_percentage_unions_axes_and_velocity_increment_cycles(tmp_path):
    from dwvp_access_metrics import command_metrics
    # One speed+increment violation, one increment-only violation, two feasible.
    (tmp_path/'raw.csv').write_text('stamp_s,vx,vy,omega\n1,.3,.3,0\n1.04,.02,0,0\n1.08,0,0,0\n1.12,0,0,0\n')
    (tmp_path/'applied.csv').write_text('stamp_s,vx,vy,omega\n.99,0,0,0\n1.03,0,0,0\n1.07,0,0,0\n1.11,0,0,0\n')
    result=command_metrics(tmp_path,{'start_stamp_s':1,'duration_s':.2},experiment.default_config()['common'])
    assert result['constraint_violation_pct']==50.
    assert result['constraint_violation_samples']==2
    assert result['constraint_evaluable_samples']==result['constraint_total_samples']==4
    assert result['constraint_unknown_samples']==0


def test_constraint_percentage_does_not_treat_missing_prior_as_feasible(tmp_path):
    from dwvp_access_metrics import command_metrics
    (tmp_path/'raw.csv').write_text('stamp_s,vx,vy,omega\n1,.3,0,0\n1.04,0,0,0\n')
    result=command_metrics(tmp_path,{'start_stamp_s':1,'duration_s':.2},experiment.default_config()['common'])
    assert result['constraint_violation_samples']==1
    assert result['constraint_evaluable_samples']==1
    assert result['constraint_unknown_samples']==1
    assert result['constraint_violation_pct'] == 100.


@pytest.mark.parametrize('bad', ['1.04,nan,0,0', 'nan,0,0,0'])
def test_constraint_percentage_invalid_raw_is_unknown(tmp_path,bad):
    from dwvp_access_metrics import command_metrics
    (tmp_path/'raw.csv').write_text('stamp_s,vx,vy,omega\n1,0,0,0\n'+bad+'\n')
    (tmp_path/'applied.csv').write_text('stamp_s,vx,vy,omega\n.99,0,0,0\n')
    result=command_metrics(tmp_path,{'start_stamp_s':1,'duration_s':.2},experiment.default_config()['common'])
    assert result['constraint_total_samples']==2
    assert result['constraint_unknown_samples']==1
    assert result['constraint_violation_pct'] == 0.


def test_experiment_profiles_keep_goal_approach_enabled(tmp_path):
    root=tmp_path/'session'
    experiment.prepare(root,ROOT/'params/hsrb_dwvp_access_params.yaml')
    params=yaml.safe_load((root/'nav2_params.yaml').read_text())['controller_server']['ros__parameters']
    for name in ('DWVP','VP_CLIP','VP_SCALED'):
        assert params[name]['approach_velocity_scaling_dist']==.6


def test_constraint_percentage_invalid_latest_applied_cannot_use_older_sample(tmp_path):
    from dwvp_access_metrics import command_metrics
    (tmp_path/'raw.csv').write_text('stamp_s,vx,vy,omega\n1,0,0,0\n')
    (tmp_path/'applied.csv').write_text('stamp_s,vx,vy,omega\n.98,0,0,0\n.99,nan,0,0\n')
    result=command_metrics(tmp_path,{'start_stamp_s':1,'duration_s':.2},experiment.default_config()['common'])
    assert result['constraint_unknown_samples']==1
    assert result['constraint_violation_pct'] is None
    assert result['vx_increment_excess_ratio_max'] is None


@pytest.mark.parametrize('method', ['VP_CLIP', 'VP_SCALED', 'DWVP'])
def test_half_acceleration_is_shared_by_controller_smoother_and_metrics(tmp_path, method):
    root, folder, name = make_attempt(tmp_path, 'E1_orientation_half', method)
    manifest, trial, _ = experiment.load_trial(root, name)
    params = yaml.safe_load((root / trial['params_file']).read_text())
    ctrl = params['controller_server']['ros__parameters'][method]
    smooth = params['velocity_smoother']['ros__parameters']
    assert [ctrl['max_accel_' + axis] for axis in ('x','y','theta')] == [.11,.11,.3]
    assert smooth['max_accel'] == [.11,.11,.3]
    assert smooth['max_decel'] == [-.11,-.11,-.3]
    (folder/'raw.csv').write_text('stamp_s,vx,vy,omega\n10.1,.005,0,0\n')
    (folder/'applied.csv').write_text('stamp_s,vx,vy,omega\n10.09,0,0,0\n')
    diag = experiment.summarize(root)['command_diagnostics'][name]
    assert diag['constraint_violation_pct'] == 100.
    assert diag['vx_increment_excess_ratio_max'] == pytest.approx(.005/(.11/30)-1)


def test_unassigned_combination_and_wrong_profile_are_rejected(tmp_path):
    root=tmp_path/'session'
    manifest=experiment.prepare(root,ROOT/'params/hsrb_dwvp_access_params.yaml',[0,0,0])
    with pytest.raises(ValueError,match='unassigned'):
        experiment.load_trial(root,'E1_lateral_MPPI_r1')
    trial = next(t for t in manifest['trials'] if t['id']=='E1_orientation_half_DWVP_r1')
    trial['params_file'] = manifest['parameter_sets']['E1_orientation_nominal']['file']
    trial['params_sha256'] = manifest['parameter_sets']['E1_orientation_nominal']['sha256']
    experiment.write_json(root/'manifest.json', manifest)
    with pytest.raises(ValueError,match='condition parameters'):
        experiment.load_trial(root,trial['id'])


def test_assignment_check_uses_frozen_config(tmp_path):
    root=tmp_path/'session'
    manifest=experiment.prepare(root,ROOT/'params/hsrb_dwvp_access_params.yaml',[0,0,0])
    trial = next(t for t in manifest['trials'] if t['id']=='E1_lateral_DWVP_r1')
    trial['controller'] = 'MPPI'
    experiment.write_json(root/'manifest.json', manifest)
    with pytest.raises(ValueError,match='Unassigned'):
        experiment.load_trial(root,trial['id'])


def test_dwb_omni_dynamic_window_limits_and_goal_checker():
    config=experiment.default_config()
    params=experiment.render_parameters(ROOT/'params/hsrb_dwvp_access_params.yaml', config)
    ctrl=params['controller_server']['ros__parameters']['DWB']
    assert ctrl['wrapped_plugin']=='dwb_core::DWBLocalPlanner'
    assert ctrl['trajectory_generator_name']=='dwb_plugins::LimitedAccelGenerator'
    assert ctrl['min_vel_y']==-.22 and ctrl['max_vel_y']==.22 and ctrl['vy_samples']>1
    assert ctrl['max_speed_xy']==pytest.approx(np.hypot(.22,.22))
    assert ctrl['sim_period']==1/30 and ctrl['xy_goal_tolerance']==.1
    for i, axis in enumerate(('x','y','theta')):
        assert ctrl['acc_lim_'+axis]==config['common']['max_accel'][i]
        assert ctrl['decel_lim_'+axis]==config['common']['max_decel'][i]


@pytest.mark.parametrize('speed,age', [(float('nan'),0.),(.2,1.)])
def test_orientation_speed_does_not_fill_missing_or_stale_values(tmp_path,speed,age):
    root,folder,_=make_attempt(tmp_path,'E1_orientation_half','DWVP',[[1.15,0,.7]])
    file=folder/'tracking.csv'
    lines=file.read_text().splitlines()
    values=lines[1].split(',')
    values[-1]=str(speed)
    values[lines[0].split(',').index('odom_source_age_s')]=str(age)
    file.write_text(lines[0]+'\n'+','.join(values)+'\n')
    row=experiment.summarize(root)['trials'][0]
    assert row['orientation_min_speed_m_s'] is None
    assert row['orientation_speed_samples']==0
