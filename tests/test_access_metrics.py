"""Analytical and simulator-to-recorder checks, executed in networkless Docker."""
import csv
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import dwvp_access_experiment as experiment
from dwvp_access_metrics import (COMMON_METRICS, command_metrics, tracking_metrics,
                                 transient_metrics, write_group_tables)
from dwvp_access_noise import noise_metrics


def lateral(times, errors, valid=None):
    config = experiment.default_config()
    poses = np.c_[np.linspace(0, 1.5, len(times)), errors, np.zeros(len(times))]
    return tracking_metrics(np.array(times), poses, experiment.canonical_path('E1_lateral'),
                            config['conditions']['E1_lateral'], [0,.5,0], valid)[0]


def test_constant_error_irregular_time_average():
    m = lateral([0,.1,1.,3.], [.2]*4)
    assert m['eval_max_position_error_m'] == pytest.approx(.2)
    assert m['eval_position_error_integral_m_s'] == pytest.approx(.6)
    assert m['eval_mean_position_error_m'] == pytest.approx(.2)
    assert m['eval_duration_s'] == 3.


def test_triangular_error_and_opposite_initial_side():
    triangle = lateral([0,1,3], [0,2,0])
    assert triangle['eval_position_error_integral_m_s'] == 3.
    assert triangle['eval_mean_position_error_m'] == 1.
    m = lateral([0,1,3], [.5,-.25,0])
    assert m['eval_position_error_integral_m_s'] == pytest.approx(.625)
    assert m['eval_mean_position_error_m'] == pytest.approx(.625/3)
    assert m['crossing_m'] == .25


def test_invalid_sample_never_bridges_an_integral():
    m = lateral([0,1,10,11], [1,1,1,1], np.array([True,False,True,True]))
    assert m['eval_duration_s'] == 1.
    assert m['eval_position_error_integral_m_s'] == 1.
    assert m['eval_mean_position_error_m'] == 1.


def test_empty_singleton_and_nonincreasing_time():
    m = lateral([0], [.2])
    assert m['eval_mean_position_error_m'] is None
    assert m['eval_position_error_integral_m_s'] == 0.
    m = lateral([0,1], [.2,.2], np.zeros(2,dtype=bool))
    assert m['eval_position_error_integral_m_s'] is None
    m = lateral([1,0,2], [.2,.2,.2])
    assert m['invalid_time_intervals'] == 1
    assert m['eval_duration_s'] == 2.


def test_window_excludes_prestart_and_terminal_rotation():
    poses = np.array([[-.1,3,2],[0,.2,.1],[1,.2,.1],[1.9,.2,.1],[2.4,4,3]])
    c = experiment.default_config()['conditions']['E1_lateral']
    m,_,mask = tracking_metrics(np.arange(5.), poses, experiment.canonical_path('E1_lateral'),c,[0,.5,0])
    assert mask.tolist() == [False,True,True,True,False]
    assert m['eval_mean_heading_error_deg'] == pytest.approx(math.degrees(.1))
    assert m['eval_heading_error_integral_deg_s'] == pytest.approx(2*math.degrees(.1))
    assert m['eval_max_position_error_m'] == .2


@pytest.mark.parametrize('rotation', [0., 1.2])
def test_lag_is_in_ramp_and_overshoot_strictly_after_ramp(rotation):
    c = experiment.default_config()['conditions']['E1_orientation_nominal']
    path = experiment.canonical_path('E1_orientation_nominal')
    poses = np.array([[.5,0,-1.],[1.,0,-.1],[1.15,0,np.pi/4-.2],
                      [1.3,0,np.pi/2+.8],[1.4,0,np.pi/2+.3],[2.,0,np.pi/2+1.]])
    origin = [2.,-3.,rotation]
    path = experiment.transform_poses(path,origin)
    poses = experiment.transform_poses(poses,origin)
    m,_,_ = tracking_metrics(np.arange(6.),poses,path,c,origin)
    assert m['transition_heading_lag_deg'] == pytest.approx(math.degrees(.2))
    assert m['post_transition_heading_overshoot_deg'] == pytest.approx(math.degrees(.3))


def test_environment_uses_projected_arc_length():
    path = np.array([[0,0,0],[1,0,0],[1,1,np.pi/2]])
    poses = np.array([[.5,.1,0],[1.1,.3,.5],[1,1,3]])
    c = experiment.default_config()['conditions']['E2_environment']
    m,projected,mask = tracking_metrics(np.array([0.,2.,3.]),poses,path,c,path[0])
    np.testing.assert_allclose(projected[:,2],[.5,1.3,2.])
    assert mask.tolist() == [True,True,False]
    assert m['evaluation_end_m'] == 1.4
    assert m['eval_position_error_integral_m_s'] == pytest.approx(.2)
    assert m['crossing_m'] is None


def test_command_unknown_bounds_and_threshold(tmp_path):
    (tmp_path/'raw.csv').write_text('stamp_s,vx,vy,omega\n1,.3,0,0\n1.04,0,0,0\n')
    m = command_metrics(tmp_path,dict(start_stamp_s=1.,duration_s=.2),experiment.default_config()['common'])
    assert m['constraint_violation_pct'] == 100.
    assert m['constraint_unknown_pct'] == 50.
    assert m['constraint_violation_lower_pct'] == 50.
    assert m['constraint_violation_upper_pct'] == 100.
    assert m['constraint_unknown_flag']
    m = command_metrics(tmp_path,dict(start_stamp_s=1.,duration_s=.2),experiment.default_config()['common'], unknown_threshold=50.)
    assert not m['constraint_unknown_flag']


def test_monotonic_pairing_ignores_ros_clock_jump(tmp_path):
    (tmp_path/'raw.csv').write_text('stamp_s,vx,vy,omega,receive_monotonic_s\n13,0,0,0,1.03\n')
    (tmp_path/'applied.csv').write_text('stamp_s,vx,vy,omega,receive_monotonic_s\n10,0,0,0,1.02\n')
    m = command_metrics(tmp_path,dict(start_stamp_s=9.,duration_s=5.),experiment.default_config()['common'])
    assert m['constraint_violation_pct'] == 0.
    assert m['constraint_unknown_samples'] == 0
    assert m['pairing_clock'] == 'receive_monotonic_s'


def test_noise_sample_sd_excursion_and_yaw_wrap():
    m = noise_metrics([[0,0,np.deg2rad(179)],[2,0,np.deg2rad(-179)]])
    assert m['position_std_m'] == pytest.approx(np.sqrt(2))
    assert m['position_max_excursion_m'] == 1.
    assert m['heading_std_deg'] == pytest.approx(np.sqrt(2))
    assert m['heading_max_excursion_deg'] == 1.
    assert m['heading_peak_to_peak_deg'] == 2.
    assert noise_metrics([])['position_std_m'] is None
    assert noise_metrics([[0,0,0]])['position_std_m'] is None


def test_table_zero_omission_requires_every_method_has_data(tmp_path):
    groups = []
    for method in ('DWPP','DWVP'):
        g = dict(task='E1_lateral',controller=method,recorded=2,succeeded=2,valid_successes=1,
                 evaluation_complete_count=2,constraint_unknown_samples=1,constraint_unknown_flagged_runs=1)
        for k in COMMON_METRICS + transient_metrics('E1_lateral') + ('constraint_unknown_pct','constraint_violation_lower_pct','constraint_violation_upper_pct'):
            g.update({k+'_mean':0.,k+'_sd':0.,k+'_n':2})
        groups.append(g)
    write_group_tables(tmp_path, groups)
    header = (tmp_path/'tables/E1_lateral.csv').read_text().splitlines()[0]
    assert 'constraint_violation_pct' not in header
    assert 'constraint_unknown_pct' in header and 'constraint_violation_upper_pct' in header
    assert 'crossing_m' in header and 'settling' not in header and 'orientation_min_speed' not in header
    assert '0% for every method over classified cycles' in (tmp_path/'tables/E1_lateral.md').read_text()
    groups[0]['recorded'] = 0
    write_group_tables(tmp_path, groups)
    assert 'constraint_violation_pct' in (tmp_path/'tables/E1_lateral.csv').read_text().splitlines()[0]
    assert '0% for every' not in (tmp_path/'tables/E1_lateral.note.txt').read_text()


def test_group_means_sample_sd_and_partial_attempts(tmp_path):
    session = tmp_path/'session'
    experiment.prepare(session,ROOT/'params/hsrb_dwvp_access_params.yaml',[0.,0.,0.])
    for repeat, error, success in ((1,.2,True),(2,.4,False)):
        folder = session/'runs'/f'E1_lateral_DWVP_r{repeat}'
        folder.mkdir(parents=True)
        np.savetxt(folder/'reference.csv',experiment.canonical_path('E1_lateral'),delimiter=',',header='x,y,yaw',comments='')
        experiment.write_json(folder/'result.json',dict(status='succeeded' if success else 'timeout',success=success,
            start_stamp_s=0.,duration_s=1.,missing_command_prefix_s=.5))
        (folder/'tracking.csv').write_text('t,stamp_s,x,y,yaw,tf_age_s,raw_age_s,applied_age_s,odom_age_s,odom_source_age_s\n'
            f'0,0,0,{error},0,0,0,0,0,0\n1,1,1,{error},0,0,0,0,0,0\n')
    report = experiment.summarize(session)
    g = next(g for g in report['groups'] if g['task']=='E1_lateral' and g['controller']=='DWVP')
    assert g['recorded']==2 and g['valid_successes']==0 and g['evaluation_complete_count']==0
    assert g['eval_mean_position_error_m_n']==2
    assert g['eval_mean_position_error_m_mean']==pytest.approx(.3)
    assert g['eval_mean_position_error_m_sd']==pytest.approx(np.sqrt(.02))
    assert g['travel_time_s_n']==1 and g['travel_time_s_sd'] is None


@pytest.mark.parametrize('index', range(8))
def test_simulator_trajectories_through_session_summary(tmp_path, index):
    assert os.environ.get('DWVP_METRICS_REFERENCE'), 'Run scripts/verify_hardware_tooling.sh to generate locked simulator references'
    reference = Path(os.environ['DWVP_METRICS_REFERENCE'])
    bundle = json.loads((reference/'reference.json').read_text())
    case = bundle['cases'][index]
    arrays = np.load(reference/case['trajectory'])
    controller = dict(vp='VP_CLIP',vp_scaled='VP_SCALED',dwvp='DWVP',dwpp='DWPP')[case['method']]
    session = tmp_path/'session'
    experiment.prepare(session,ROOT/'params/hsrb_dwvp_access_params.yaml',[0.,0.,0.])
    trial_id = case['condition']+'_'+controller+'_r1'
    folder = session/'runs'/trial_id
    folder.mkdir(parents=True)
    np.savetxt(folder/'reference.csv',arrays['path'],delimiter=',',header='x,y,yaw',comments='')
    expected = case['metrics']
    experiment.write_json(folder/'result.json',dict(status='succeeded' if expected['success'] else 'timeout',
        success=expected['success'],start_stamp_s=0.,duration_s=expected['duration_s'],missing_command_prefix_s=0.))
    with (folder/'tracking.csv').open('w') as f:
        w = csv.writer(f)
        w.writerow(['t','stamp_s','x','y','yaw','tf_age_s','raw_age_s','applied_age_s','odom_age_s','odom_source_age_s','speed_m_s'])
        for t,p in zip(arrays['times'],arrays['poses']):
            w.writerow([t,t,*p,0,0,0,0,0,0.])
    dt = 1/case['config']['frequency']
    for name, values, stamps in (
        ('raw',arrays['commands'],arrays['times'][:-1]),
        ('applied',np.vstack((np.zeros(3),arrays['applied'])),np.r_[-dt/2,arrays['times'][:-1]+dt/2])):
        with (folder/(name+'.csv')).open('w') as f:
            w=csv.writer(f);w.writerow(['stamp_s','vx','vy','omega'])
            w.writerows([[t,*v] for t,v in zip(stamps,values)])
    report = experiment.summarize(session)
    observed = report['trials'][0]
    assert not observed['data_errors']
    keys = COMMON_METRICS[:-1] + transient_metrics(case['condition']) + ('eval_duration_s',)
    target = reference/'comparison.csv'
    exists = target.exists()
    with target.open('a') as f:
        w=csv.writer(f)
        if not exists:w.writerow(['condition','method','metric','simulator','tooling','absolute_difference','absolute_tolerance'])
        for key in keys:
            original = 'command_constraint_violation_pct' if key=='constraint_violation_pct' else key
            assert observed[key] == pytest.approx(expected[original],abs=1e-9,rel=0), (key,observed[key],expected[original])
            w.writerow([case['condition'],controller,key,expected[original],observed[key],abs(observed[key]-expected[original]),1e-9])
    if case['condition'].startswith('E1_orientation'):
        assert (folder/'orientation_timeseries.csv').exists()
    # A constraint-quality flag must not remove an otherwise finite observation.
    (folder/'applied.csv').write_text('stamp_s,vx,vy,omega\n')
    report = experiment.summarize(session)
    g = next(g for g in report['groups'] if g['task']==case['condition'] and g['controller']==controller)
    assert report['trials'][0]['constraint_unknown_flag']
    assert g['eval_mean_position_error_m_n'] == 1 and g['travel_time_s_n'] == 1
