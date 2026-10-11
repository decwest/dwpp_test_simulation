"""E1a schedule, frozen tuning, measured response and simulator correspondence."""
from collections import Counter
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import dwvp_access_experiment as experiment
import dwvp_access_batch as batch
from dwvp_access_metrics import lateral_response_metrics, read_events

PARAMS = ROOT/'params/hsrb_dwvp_access_params.yaml'


def test_new_default_and_e1_schedules_and_fixed_response_parameters(tmp_path):
    full = experiment.prepare(tmp_path/'full', PARAMS, origin=[0,0,0])
    assert len(full['trials']) == 90
    session = tmp_path/'e1'
    one = experiment.prepare(session, PARAMS, current_start=[3,4,1.2],
                             conditions=['E1'], bidirectional=True)
    two = experiment.prepare(tmp_path/'two', PARAMS, current_start=[3,4,1.2],
                             conditions=['E1'], bidirectional=True)
    assert one['trials'] == two['trials']
    assert len(one['trials']) == 65
    assert Counter(t['task'] for t in one['trials']) == {
        **{k: 5 for k in experiment.RESPONSE_CONDITIONS},
        **{k: 15 for k in experiment.LEGACY_CONDITIONS[1:-1]}}
    assert batch.selected_trials(session, conditions=['E1'])[1] == one['trials']
    assert [t['direction'] for t in one['trials']] == ['forward' if i%2==0 else 'reverse' for i in range(65)]
    for repeat in range(1,6):
        block = [t for t in one['trials'] if t['repeat']==repeat]
        assert list(dict.fromkeys(t['task'] for t in block)) == list(experiment.E1_CONDITIONS)
    for name, look, tau in zip(experiment.RESPONSE_CONDITIONS, [.11,.11,.33,.33], [.5,1.5,.5,1.5]):
        trial = next(t for t in one['trials'] if t['task']==name)
        assert trial['controller']=='DWVP'
        _, _, path = experiment.load_trial(session, trial['id'])
        assert experiment.arclength(path[:,:2])[-1] == pytest.approx(2.5)
        pose = experiment.start_pose(one,trial)
        assert np.linalg.norm(pose[:2]-path[0,:2]) == pytest.approx(.5)
        params = yaml.safe_load((session/trial['params_file']).read_text())
        ctrl = params['controller_server']['ros__parameters']
        assert ctrl['DWVP']['min_lookahead_dist']==ctrl['DWVP']['max_lookahead_dist']==look
        assert ctrl['DWVP']['lateral_time_constant']==tau
        assert ctrl['VP_CLIP']['lateral_time_constant']==ctrl['VP_SCALED']['lateral_time_constant']==0.
    original = yaml.safe_load((session/'nav2_params_E1_orientation_nominal.yaml').read_text())
    assert all(original['controller_server']['ros__parameters'][c]['lateral_time_constant']==0.
               for c in ('DWVP','VP_CLIP','VP_SCALED'))


@pytest.mark.parametrize('missing', [False, True])
def test_runtime_rejects_missing_or_disabled_compensation(tmp_path, missing):
    params = experiment.render_parameters(PARAMS, experiment.default_config(), experiment.RESPONSE_CONDITIONS[1])
    expected = tmp_path/'expected.yaml'; expected.write_text(yaml.safe_dump(params))
    files = {}
    for name in ('controller_server','velocity_smoother'):
        files[name]=tmp_path/(name+'.yaml')
        files[name].write_text(yaml.safe_dump({'/'+name:params[name]}))
    experiment.verify_runtime_parameters(expected,files,'DWVP')
    ctrl=params['controller_server']['ros__parameters']['DWVP']
    if missing: ctrl.pop('lateral_time_constant')
    else: ctrl['lateral_time_constant']=0.
    files['controller_server'].write_text(yaml.safe_dump({'/controller_server':params['controller_server']}))
    with pytest.raises(ValueError,match='DWVP.lateral_time_constant'):
        experiment.verify_runtime_parameters(expected,files,'DWVP')


@pytest.mark.parametrize('tau', [.5,1.5])
def test_exponential_fit_uses_action_clock_and_records_window(tau):
    times=np.arange(0,15,1/30)
    errors=.5*np.exp(-times/tau)
    result=lateral_response_metrics(times,errors,complete=True)
    assert result['tau_fit_s']==pytest.approx(tau)
    for pct in (2,5):
        assert result[f'settling_{pct}pct_time_s']==pytest.approx(-tau*np.log(pct/100),abs=1/30)
    selected=(errors>.02)&(errors<.15)&(times<20)
    assert result['tau_fit_samples']==selected.sum()
    assert result['tau_fit_start_s']==times[selected][0]
    assert result['tau_fit_end_s']==times[selected][-1]
    assert result['tau_fit_r_squared']==pytest.approx(1.)


def test_settling_reentry_missing_tail_failure_and_insufficient_fit():
    times=np.arange(8.)
    errors=[.5,.009,.008,.03,.007,.006,.005,.004]
    result=lateral_response_metrics(times,errors,complete=True)
    assert result['settling_2pct_time_s']==4.
    assert result['settling_5pct_time_s']==4.
    assert result['tau_fit_s'] is None
    assert lateral_response_metrics(times,errors,complete=False)['settling_2pct_time_s'] is None
    assert lateral_response_metrics(times,errors,[True]*7+[False],complete=True)['settling_2pct_time_s'] is None
    assert lateral_response_metrics(times,errors,[True]*6+[False,True],complete=True)['settling_2pct_time_s'] is None
    assert lateral_response_metrics(times[::-1],errors,complete=True)['tau_fit_s'] is None
    assert lateral_response_metrics(times+20,np.linspace(.14,.03,8))['tau_fit_samples']==0
    assert lateral_response_metrics(times,np.linspace(.03,.14,8))['tau_fit_s'] is None


def test_appended_plugin_csv_columns_do_not_change_named_numeric_fields(tmp_path):
    plain='stamp_s,x,y,yaw,vx_cmd,vy_cmd,w_cmd\n0,0,.5,0,0,-.1,0\n1,.1,.4,0,.1,-.1,0\n'
    first=tmp_path/'plain.csv';first.write_text(plain)
    lines=plain.splitlines()
    extra=tmp_path/'extra.csv'
    extra.write_text(lines[0]+',response_rule,lateral_error,chord_length,response_distance\n'
                     +lines[1]+',chord,.5,.11,.33\n'+lines[2]+',inactive,nan,nan,nan\n')
    a,_=read_events(first,('stamp_s','x','y','yaw'))
    b,error=read_events(extra,('stamp_s','x','y','yaw'))
    assert error is None
    for field in a.dtype.names: np.testing.assert_array_equal(a[field],b[field])
    from benchmark_compare_three_methods_from_nav2_csv import load_nav2_csv_result
    # Legacy plugin-log analysis also uses DictReader by name.
    a=load_nav2_csv_result(first, "map");b=load_nav2_csv_result(extra, "map")
    np.testing.assert_array_equal(a.poses,b.poses)


def test_locked_simulator_counterparts_through_robot_summary(tmp_path):
    reference=Path(os.environ['DWVP_RESPONSE_REFERENCE'])
    bundle=json.loads((reference/'report.json').read_text())
    session=tmp_path/'session'
    manifest=experiment.prepare(session,PARAMS,origin=[0,0,0],conditions=['E1'],repeats=[1])
    for case in bundle['trials']:
        arrays=np.load(reference/case['trial_id']/'trajectory.npz')
        folder=session/'runs'/case['trial_id'];folder.mkdir(parents=True)
        np.savetxt(folder/'reference.csv',arrays['path'],delimiter=',',header='x,y,yaw',comments='')
        experiment.write_json(folder/'result.json',dict(status='succeeded',success=True,
            duration_s=case['duration_s'],start_stamp_s=0.,missing_command_prefix_s=0.))
        with (folder/'tracking.csv').open('w') as stream:
            writer=csv.writer(stream)
            writer.writerow(['t','stamp_s','x','y','yaw','tf_age_s','raw_age_s','applied_age_s','odom_age_s','odom_source_age_s'])
            writer.writerows([t,t,*pose,0,0,0,0,0] for t,pose in zip(arrays['times'],arrays['poses']))
    report=experiment.summarize(session)
    assert len(report['trials'])==4
    for row, expected in zip(sorted(report['trials'],key=lambda x:x['trial_id']),sorted(bundle['trials'],key=lambda x:x['trial_id'])):
        assert not row['data_errors']
        for key in ('tau_fit_s','settling_2pct_time_s','settling_5pct_time_s','crossing_m','lateral_convergence_time_s'):
            assert row[key]==pytest.approx(expected[key],abs=1e-9)
        assert row['tau_fit_samples']==expected['tau_fit_samples']
        assert (session/'tables'/(row['task']+'.csv')).exists()
    with (session/'trial_metrics.csv').open() as stream:
        assert {r['task'] for r in csv.DictReader(stream)}==set(experiment.RESPONSE_CONDITIONS)
