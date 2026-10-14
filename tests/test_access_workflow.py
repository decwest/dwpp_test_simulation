"""Artifact selection and process ownership for the one-terminal entry points."""
from datetime import datetime
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import time

import numpy as np
from PIL import Image
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import dwvp_access_workflow as workflow


def make_map(folder):
    folder.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.array([[254,205], [0,254]], dtype=np.uint8)).save(folder/'map.pgm')
    (folder/'map.yaml').write_text(yaml.safe_dump(dict(image='map.pgm', resolution=.05,
        origin=[0,0,0], free_thresh=.196, occupied_thresh=.65, mode='trinary', negate=0)))
    return folder/'map.yaml'


def test_jst_names_collisions_and_existing_environment(monkeypatch, tmp_path):
    class Clock:
        @staticmethod
        def now(zone):
            assert zone.key == 'Asia/Tokyo'
            return datetime(2026,10,7,0,1,2)
    monkeypatch.setattr(workflow, 'datetime', Clock)
    assert workflow.new_directory(tmp_path, 'lab').name == 'lab_20261007_000102'
    assert workflow.new_directory(tmp_path, 'lab').name == 'lab_20261007_000102_01'
    with pytest.raises(FileExistsError):
        workflow.new_directory(tmp_path, 'lab', 'lab_20261007_000102')
    with pytest.raises(ValueError):
        workflow.new_directory(tmp_path, 'lab', '../old')


def test_latest_is_explicit_and_experiment_map_is_a_snapshot(tmp_path):
    maps = tmp_path/'maps'
    saved = make_map(maps/'lab_saved')
    make_map(maps/'lab_partial')  # Never select by mtime or a failed run's directory name.
    workflow.atomic_json(maps/'latest.json', dict(map='lab_saved/map.yaml'))
    assert workflow.choose_map(tmp_path) == saved
    copy = workflow.copy_map(saved, tmp_path/'snapshot')
    image = copy.with_suffix('.pgm').read_bytes()
    saved.with_suffix('.pgm').write_bytes(b'changed after the run')
    assert copy.with_suffix('.pgm').read_bytes() == image
    assert workflow.checked_map(copy)['free_thresh'] == .196


def test_old_unknown_threshold_is_rejected(tmp_path):
    saved = make_map(tmp_path/'map')
    info = yaml.safe_load(saved.read_text()); info['free_thresh'] = .25
    saved.write_text(yaml.safe_dump(info))
    with pytest.raises(ValueError, match='Unknown gray'):
        workflow.checked_map(saved)


class FakeProcess:
    instances = []
    def __init__(self, command, logfile):
        self.command, self.logfile, self.stopped = command, logfile, False
        self.instances.append(self)
    def __enter__(self): return self
    def __exit__(self, *args): self.stop()
    def stop(self): self.stopped = True
    def check(self): assert not self.stopped


def fake_operator(monkeypatch):
    FakeProcess.instances = []
    monkeypatch.setattr(workflow, 'Process', FakeProcess)
    monkeypatch.setattr(workflow, 'wait_enter', lambda message, background: workflow.check_background(background))
    def stopped():
        assert FakeProcess.instances[-1].stopped  # Joy must close before capture/save/motion.
        return dict(pose=[0.,0.,0.])
    monkeypatch.setattr(workflow, 'stopped_pose', stopped)


@pytest.mark.parametrize('fail', [False, True])
def test_map_save_failure_keeps_previous_selection_and_stops_only_owned_nodes(monkeypatch, tmp_path, fail):
    fake_operator(monkeypatch)
    old = make_map(tmp_path/'maps/old')
    pointer = old.parent.parent/'latest.json'
    workflow.atomic_json(pointer, dict(map='old/map.yaml'))
    old_bytes = pointer.read_bytes()
    folder = tmp_path/'maps/new'; folder.mkdir()
    def save(command, logfile, background, timeout):
        assert FakeProcess.instances[-1].stopped
        assert not FakeProcess.instances[0].stopped
        assert command[command.index('--free')+1] == '0.196'
        if fail: raise RuntimeError('save failed')
        make_map(folder)
    monkeypatch.setattr(workflow, 'run_child', save)
    args = NS(no_rviz=False, no_joy=False)
    if fail:
        entered = []
        def stop_retry(message, background):
            workflow.check_background(background)
            entered.append(message)
            if len(entered)>1:
                assert not FakeProcess.instances[0].stopped  # Keep SLAM alive for a retry.
                raise KeyboardInterrupt()
        monkeypatch.setattr(workflow, 'wait_enter', stop_retry)
        with pytest.raises(KeyboardInterrupt):
            workflow.mapping(args, folder)
        assert pointer.read_bytes() == old_bytes
    else:
        workflow.mapping(args, folder)
        assert workflow.choose_map(tmp_path) == folder/'map.yaml'
        assert (folder/'environment.json').exists()
    assert all(p.stopped for p in FakeProcess.instances)


def test_failed_map_save_can_retry_without_restarting_slam(monkeypatch, tmp_path):
    fake_operator(monkeypatch)
    folder = tmp_path/'maps/new'; folder.mkdir(parents=True)
    attempts = []
    def save(command, logfile, background, timeout):
        workflow.check_background(background)
        attempts.append(logfile.name)
        if len(attempts)==1: raise RuntimeError('Temporary map timeout')
        make_map(folder)
    monkeypatch.setattr(workflow, 'run_child', save)
    workflow.mapping(NS(no_rviz=False, no_joy=False), folder)
    assert attempts == ['save_01.log', 'save_02.log']
    assert len(FakeProcess.instances)==2 and all(p.stopped for p in FakeProcess.instances)
    assert workflow.choose_map(tmp_path)==folder/'map.yaml'


@pytest.mark.parametrize('fail', [False, True])
@pytest.mark.parametrize('noise_duration', [None, 2.])
def test_experiment_handoff_fresh_session_and_failure_cleanup(monkeypatch, tmp_path, fail, noise_duration):
    fake_operator(monkeypatch)
    source = make_map(tmp_path/'maps/lab_test')
    folder = tmp_path/'results/dwvp_access/lab_test/run_1'; folder.mkdir(parents=True)
    args = NS(workspace=tmp_path, no_rviz=False, no_joy=False, params=ROOT/'params/hsrb_dwvp_access_params.yaml',
              config=None, repeats=None, conditions=['E1'], stationary_noise=noise_duration)
    calls = []
    def child(command, logfile, background, timeout=None):
        assert FakeProcess.instances[-1].stopped and not FakeProcess.instances[0].stopped
        if 'stationary-noise' in command:
            assert not calls and not (folder/'session').exists()
            fake_noise(command)
            return
        calls.append(command)
        if 'prepare' in command:
            workflow.experiment.prepare(folder/'session', args.params, current_start=[0,0,0], bidirectional=True, conditions=['E1'])
        elif fail and '--dry-run' not in command:
            raise RuntimeError('batch failed')
    monkeypatch.setattr(workflow, 'run_child', child)
    if fail:
        with pytest.raises(RuntimeError, match='batch failed'):
            workflow.run_experiment(args, folder, source)
    else:
        workflow.run_experiment(args, folder, source)
    manifest = json.loads((folder/'session/manifest.json').read_text())
    assert len(manifest['trials']) == 65
    assert '--bidirectional' in calls[0] and '--ordered' not in calls[0]
    assert '--start-from-current' in calls[-1] and '--resume' not in calls[-1]
    assert '--continue-on-endpoint-failure' in calls[-1]
    assert '--dry-run' in calls[-2]
    assert (folder/'map/map.pgm').read_bytes() == source.with_suffix('.pgm').read_bytes()
    assert all(p.stopped for p in FakeProcess.instances)


def fake_noise(command):
    assert command[4] == 'stationary-noise'
    assert command[command.index('--frequency')+1] == '30'
    assert command[command.index('--frame')+1] == 'map'
    assert FakeProcess.instances[-1].stopped and not FakeProcess.instances[0].stopped
    output = Path(command[command.index('--output')+1]); output.mkdir()
    workflow.experiment.write_json(output/'noise_summary.json',
        dict(x_std_m=.003, y_std_m=.004, heading_std_deg=.2, stationary_verified=True))


@pytest.mark.parametrize('noise_args,seconds', [([], None), (['--stationary-noise'], 30.),
    (['--stationary-noise', '2.5'], 2.5)])
@pytest.mark.parametrize('ordered', [False, True])
def test_one_way_cli_prepares_once_and_returns_to_fixed_start(monkeypatch, tmp_path, capsys, noise_args, seconds, ordered):
    fake_operator(monkeypatch)
    source = make_map(tmp_path/'maps/lab_test')
    workflow.atomic_json(source.parent.parent/'latest.json', dict(map='lab_test/map.yaml'))
    monkeypatch.setitem(sys.modules, 'ament_index_python.packages',
                        NS(get_package_share_directory=lambda name: str(ROOT)))
    monkeypatch.setattr(sys.stdin, 'isatty', lambda: True)
    monkeypatch.setattr(workflow, 'require_idle', lambda args: None)
    prompts, calls = [], []
    monkeypatch.setattr(workflow, 'wait_enter', lambda message, background: prompts.append(message))
    def child(command, logfile, background, timeout=None):
        assert prompts and FakeProcess.instances[-1].stopped
        workflow.check_background(background)
        calls.append(command)
        if 'stationary-noise' in command:
            assert len(calls) == 1 and not (logfile.parent/'session').exists()
            assert float(command[command.index('--duration')+1]) == seconds
            assert timeout > seconds
            fake_noise(command)
        elif 'prepare' in command:
            assert '--bidirectional' not in command and '--start-from-current' in command
            assert ('--ordered' in command) == ordered
            assert command[command.index('--repeats')+1:] == ['1']
            assert command[command.index('--conditions')+1] == 'E1'
            workflow.experiment.prepare(Path(command[command.index('--output')+1]),
                ROOT/'params/hsrb_dwvp_access_params.yaml', current_start=[0, 0, 0],
                conditions=['E1'], repeats=[1], ordered=ordered)
        else:
            assert '--start-from-current' not in command
            assert '--resume' in command  # Align to the frozen first start if necessary.
            assert '--continue-on-endpoint-failure' in command
    monkeypatch.setattr(workflow, 'run_child', child)
    monkeypatch.setattr(sys, 'argv', ['workflow', '--workspace', str(tmp_path), 'experiment', '--one-way']
        + (['--ordered'] if ordered else []) + noise_args)
    workflow.main()
    assert len(calls) == (4 if seconds is not None else 3)
    assert '--dry-run' in calls[-2] and '--dry-run' not in calls[-1]
    session = tmp_path/json.loads((tmp_path/'results/dwvp_access/latest.json').read_text())['session']
    manifest = json.loads((session/'manifest.json').read_text())
    assert len(manifest['trials']) == 13 and not manifest.get('bidirectional')
    if seconds is not None:
        assert f'Enter 後 {seconds:g} 秒は静止' in prompts[0]
        assert 'SD x=0.003000 m, y=0.004000 m, yaw=0.200000 deg' in capsys.readouterr().out
        assert (session.parent/'stationary_noise/noise_summary.json').exists()
    assert all(p.stopped for p in FakeProcess.instances)


@pytest.mark.parametrize('extra', [['--resume'], ['--retry-failed'], ['--prepared-session', 'missing'],
    ['--retry-trial', 'trial', '--source-session', 'missing']])
def test_one_way_rejects_existing_session_modes_before_side_effects(monkeypatch, tmp_path, extra):
    monkeypatch.setattr(sys, 'argv', ['workflow', '--workspace', str(tmp_path), 'experiment', '--one-way']+extra)
    with pytest.raises(SystemExit) as error:
        workflow.main()
    assert error.value.code == 2 and not list(tmp_path.iterdir())


@pytest.mark.parametrize('extra', [['--ordered'], ['--stationary-noise', '0'],
    ['--stationary-noise', '-1'], ['--stationary-noise', 'nan'], ['--stationary-noise', 'inf']])
def test_invalid_video_options_rejected_before_side_effects(monkeypatch, tmp_path, extra):
    monkeypatch.setattr(sys, 'argv', ['workflow', '--workspace', str(tmp_path), 'experiment']+extra)
    with pytest.raises(SystemExit) as error:
        workflow.main()
    assert error.value.code == 2 and not list(tmp_path.iterdir())


def test_noise_failure_prevents_prepare_batch_and_pointer_update(monkeypatch, tmp_path):
    fake_operator(monkeypatch)
    source = make_map(tmp_path/'maps/lab_test')
    folder = tmp_path/'run'; folder.mkdir()
    args = NS(workspace=tmp_path, no_rviz=False, no_joy=False,
              params=ROOT/'params/hsrb_dwvp_access_params.yaml', stationary_noise=30.)
    calls = []
    def child(command, logfile, background, timeout=None):
        calls.append(command)
        fake_noise(command)
        raise RuntimeError('Stationarity was not verified')
    monkeypatch.setattr(workflow, 'run_child', child)
    with pytest.raises(RuntimeError, match='Stationarity'):
        workflow.run_experiment(args, folder, source)
    assert len(calls) == 1 and not (folder/'session').exists()
    assert not (tmp_path/'results/dwvp_access/latest.json').exists()
    assert all(p.stopped for p in FakeProcess.instances)


@pytest.mark.parametrize('explicit_trial', [False, True])
def test_e1_retry_noise_precedes_freezing_new_attempt(monkeypatch, tmp_path, explicit_trial):
    from test_access_batch import write_success, endpoint_miss
    fake_operator(monkeypatch)
    parent = tmp_path/'results/dwvp_access/lab/run_old'
    saved_map = make_map(parent/'map')
    source = parent/'session'
    manifest = workflow.experiment.prepare(source, ROOT/'params/hsrb_dwvp_access_params.yaml',
        current_start=[0, 0, 0], bidirectional=True, conditions=['E1_response_L0p11_T0p5'], repeats=[1])
    trial = manifest['trials'][0]
    endpoint_miss(write_success(source, trial))
    before = {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
    folder = workflow.new_directory(parent/'retries', 'retry')
    args = NS(workspace=tmp_path, no_rviz=False, no_joy=False, retry_source=source,
        retry_failed=None if explicit_trial else source, retry_trial=trial['id'] if explicit_trial else None,
        params=source/trial['params_file'], stationary_noise=2.)
    calls = []
    def child(command, logfile, background, timeout=None):
        calls.append(command)
        if 'stationary-noise' in command:
            assert len(calls) == 1 and not (folder/'session').exists()
            fake_noise(command)
        else:
            assert (folder/'stationary_noise/noise_summary.json').exists()
            assert '--start-from-current' in command and '--resume' not in command
    monkeypatch.setattr(workflow, 'run_child', child)
    workflow.run_experiment(args, folder, saved_map)
    assert len(calls) == 3
    new = json.loads((folder/'session/manifest.json').read_text())
    assert 'retry_of' in new and new['trials'] == manifest['trials']
    assert all(p.read_bytes() == data for p, data in before.items())


def prepare_resume(workspace):
    from test_access_batch import write_success, endpoint_miss
    parent = workspace/'results/dwvp_access/lab_saved/run_old'
    saved_map = make_map(parent/'map')
    session = parent/'session'
    manifest = workflow.experiment.prepare(session, ROOT/'params/hsrb_dwvp_access_params.yaml',
        current_start=[0,0,0], bidirectional=True, conditions=['E1'], repeats=[1])
    write_success(session, manifest['trials'][0])
    endpoint_miss(write_success(session, manifest['trials'][1]), legacy=True)
    batch = session/'batches/previous'; batch.mkdir(parents=True)
    workflow.experiment.write_json(batch/'map_input.json', dict(yaml_sha256=workflow.experiment.digest(saved_map),
        image_sha256=workflow.experiment.digest(saved_map.with_suffix('.pgm'))))
    workflow.atomic_json(workspace/'results/dwvp_access/latest.json', dict(session=str(session.relative_to(workspace))))
    return session, saved_map, manifest


def test_resume_selects_original_map_and_frozen_pending_parameters(tmp_path):
    session, saved_map, manifest = prepare_resume(tmp_path)
    # The newest mapping environment must not replace this session's map.
    new = make_map(tmp_path/'maps/new_environment')
    workflow.atomic_json(new.parent.parent/'latest.json', dict(map='new_environment/map.yaml'))
    actual, source, pending = workflow.resume_inputs(tmp_path, 'latest')
    assert actual == session and source == saved_map and pending == manifest['trials'][2:]
    assert workflow.resume_inputs(tmp_path, session)[0] == session
    saved_map.with_suffix('.pgm').write_bytes(new.with_suffix('.pgm').read_bytes()+b'changed')
    with pytest.raises(ValueError, match='Saved map changed'):
        workflow.resume_inputs(tmp_path, session)


@pytest.mark.parametrize('noise_duration', [None, 2.])
def test_resume_never_prepares_or_rewrites_previous_trials(monkeypatch, tmp_path, noise_duration):
    fake_operator(monkeypatch)
    session, saved_map, manifest = prepare_resume(tmp_path)
    _, _, pending = workflow.resume_inputs(tmp_path, session)
    before = {str(p): p.read_bytes() for p in session.rglob('*') if p.is_file()}
    args = NS(workspace=tmp_path, no_rviz=False, no_joy=False, resume=session,
              resume_session=session, params=session/pending[0]['params_file'], stationary_noise=noise_duration)
    folder = workflow.new_directory(session.parent/'resumes', 'resume')
    calls = []
    def child(command, logfile, background, timeout=None):
        if 'stationary-noise' in command:
            assert not calls
            fake_noise(command)
            return
        assert FakeProcess.instances[-1].stopped
        assert '--resume' in command and '--continue-on-endpoint-failure' in command
        assert 'prepare' not in command
        calls.append(command)
    monkeypatch.setattr(workflow, 'run_child', child)
    # Summarize regenerates derived metrics only; preserve all original raw files.
    workflow.run_experiment(args, folder, saved_map)
    assert len(calls) == 2 and '--dry-run' in calls[0]
    for path, data in before.items(): assert Path(path).read_bytes() == data
    assert str(saved_map) in calls[-1]
    assert all(p.stopped for p in FakeProcess.instances)


def test_resume_dry_run_and_input_rejection_do_not_touch_ros(monkeypatch, tmp_path, capsys):
    session, saved_map, manifest = prepare_resume(tmp_path)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    def forbidden(*args): raise AssertionError('No ROS or processes in dry run')
    monkeypatch.setattr(workflow, 'require_idle', forbidden)
    base = ['workflow', '--workspace', str(tmp_path), 'experiment', '--resume']
    monkeypatch.setattr(sys, 'argv', base+['--dry-run'])
    workflow.main()
    text = capsys.readouterr().out
    assert 'Remaining trials: 11' in text and manifest['trials'][2]['id'] in text
    assert {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()} == before
    monkeypatch.setattr(sys, 'argv', base+['--map', str(saved_map), '--dry-run'])
    with pytest.raises(ValueError, match='frozen conditions'):
        workflow.main()


def prepare_environment(workspace, bidirectional=True):
    parent = workspace/'results/dwvp_access/lab_e2/run_e2'
    saved_map = make_map(parent/'map')
    Image.fromarray(np.full((80, 100), 254, dtype=np.uint8)).save(saved_map.with_suffix('.pgm'))
    route = parent/'route'; route.mkdir()
    csv = route/'E2_environment.csv'
    np.savetxt(csv, [[1., 1., 0.], [2., 1., 0.]], delimiter=',', header='x,y,yaw', comments='')
    workflow.atomic_json(route/'path_metadata.json', dict(start=[1.,1.,0.],
        csv_sha256=workflow.experiment.digest(csv), map_yaml_sha256=workflow.experiment.digest(saved_map),
        map_image_sha256=workflow.experiment.digest(saved_map.with_suffix('.pgm'))))
    session = parent/'session'
    manifest = workflow.experiment.prepare(session, ROOT/'params/hsrb_dwvp_access_params.yaml',
        origin=[0,0,0], environment_path=csv, bidirectional=bidirectional, conditions=['E2_environment'])
    return session, saved_map, manifest


@pytest.mark.parametrize('bidirectional', [False, True])
def test_prepared_e2_dry_run_checks_frozen_route_and_map_without_ros(monkeypatch, tmp_path, capsys, bidirectional):
    session, saved_map, manifest = prepare_environment(tmp_path, bidirectional)
    def forbidden(*args): raise AssertionError('Must not start ROS, SSH or processes')
    monkeypatch.setattr(workflow, 'require_idle', forbidden)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    base = ['workflow', '--workspace', str(tmp_path), 'experiment', '--prepared-session', str(session)]
    monkeypatch.setattr(sys, 'argv', base+['--dry-run'])
    workflow.main()
    output = capsys.readouterr().out
    assert 'Trials: 25' in output and 'Fixed E2 route checked' in output
    assert {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()} == before
    monkeypatch.setattr(sys, 'argv', base+['--map', str(saved_map), '--dry-run'])
    with pytest.raises(ValueError, match='frozen conditions'):
        workflow.main()
    # A changed map must fail even when the route would still fit in free space.
    saved_map.with_suffix('.pgm').write_bytes(saved_map.with_suffix('.pgm').read_bytes()+b'changed')
    monkeypatch.setattr(sys, 'argv', base+['--dry-run'])
    with pytest.raises(ValueError, match='planner metadata'):
        workflow.main()


@pytest.mark.parametrize('resuming', [False, True])
@pytest.mark.parametrize('bidirectional', [False, True])
@pytest.mark.parametrize('noise_duration', [None, 2.])
def test_e2_workflow_previews_fixed_route_and_never_reanchors(monkeypatch, tmp_path, resuming, bidirectional, noise_duration):
    from test_access_batch import write_success
    fake_operator(monkeypatch)
    session, saved_map, manifest = prepare_environment(tmp_path, bidirectional)
    if resuming:
        write_success(session, manifest['trials'][0])
        _, _, pending = workflow.resume_inputs(tmp_path, session)
        assert pending == manifest['trials'][1:]
        with pytest.raises(FileExistsError):
            workflow.prepared_inputs(tmp_path, session)
    else:
        _, _, pending = workflow.prepared_inputs(tmp_path, session)
        assert len(pending) == 25
    before = {str(p): p.read_bytes() for p in session.rglob('*') if p.is_file()}
    args = NS(workspace=tmp_path, no_rviz=False, no_joy=False,
              prepared_session=None if resuming else session, resume=session if resuming else None,
              resume_session=session, params=session/pending[0]['params_file'], stationary_noise=noise_duration)
    folder = workflow.new_directory(session.parent/'executions', 'start')
    calls = []
    def child(command, logfile, background, timeout=None):
        if 'stationary-noise' in command:
            assert not calls and FakeProcess.instances[1].stopped
            fake_noise(command)
            return
        assert '--start-from-current' not in command and 'prepare' not in command
        assert '--resume' in command  # Includes the initial fixed-start alignment.
        assert FakeProcess.instances[1].stopped and FakeProcess.instances[2].stopped
        assert not FakeProcess.instances[0].stopped
        calls.append(command)
    monkeypatch.setattr(workflow, 'run_child', child)
    workflow.run_experiment(args, folder, saved_map)
    assert len(calls) == 2 and '--dry-run' in calls[0]
    assert 'preview' in FakeProcess.instances[1].command
    assert pending[0]['id'] in FakeProcess.instances[1].command
    for path, data in before.items(): assert Path(path).read_bytes() == data
    assert all(p.stopped for p in FakeProcess.instances)


def test_e2_current_pose_retry_rejected_before_any_side_effect(monkeypatch, tmp_path):
    session, _, _ = prepare_environment(tmp_path)
    def forbidden(*args): raise AssertionError('Must not touch robot or create retry files')
    monkeypatch.setattr(workflow, 'require_idle', forbidden)
    monkeypatch.setattr(sys, 'argv', ['workflow', '--workspace', str(tmp_path),
        'experiment', '--retry-failed', str(session), '--dry-run'])
    with pytest.raises(ValueError, match='E1-only'):
        workflow.main()
    assert not (session.parent/'retries').exists()


def interrupted_environment(workspace):
    from test_access_batch import write_success
    session, saved_map, manifest = prepare_environment(workspace, bidirectional=False)
    trial = next(t for t in manifest['trials'] if t['controller']=='DWB')
    write_success(session, trial)
    result=session/'runs'/trial['id']/'result.json'
    result.write_text(json.dumps(dict(status='interrupted',success=False,cancellation_confirmed=True)))
    return session,saved_map,trial


def test_interrupted_fixed_retry_dry_run_preserves_all_originals(monkeypatch,tmp_path,capsys):
    session,_,trial=interrupted_environment(tmp_path)
    before={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    def forbidden(*args):
        raise AssertionError('Dry run must not start ROS or stop robot teleop')
    monkeypatch.setattr(workflow,'require_idle',forbidden)
    monkeypatch.setattr(sys,'argv',['workflow','--workspace',str(tmp_path),'experiment',
        '--retry-trial',trial['id'],'--source-session',str(session),'--dry-run'])
    workflow.main()
    assert 'same frozen S -> G' in capsys.readouterr().out
    assert {str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}==before


@pytest.mark.parametrize('noise_duration', [None, 2.])
def test_fixed_retry_owns_a_new_single_trial_and_keeps_map_binding(monkeypatch,tmp_path,noise_duration):
    fake_operator(monkeypatch)
    session,saved_map,trial=interrupted_environment(tmp_path)
    workflow.fixed_retry_inputs(tmp_path,session,trial['id'])
    before={str(p):p.read_bytes() for p in session.rglob('*') if p.is_file()}
    folder=workflow.new_directory(session.parent/'retries','retry')
    args=NS(workspace=tmp_path,no_rviz=False,no_joy=False,retry_trial=trial['id'],
            retry_source=session,fixed_retry=True,params=session/trial['params_file'],stationary_noise=noise_duration)
    calls=[]
    def child(command,logfile,background,timeout=None):
        if 'stationary-noise' in command:
            assert not calls and not (folder/'session').exists()
            fake_noise(command)
            return
        assert '--start-from-current' not in command and '--resume' in command
        assert FakeProcess.instances[1].stopped and FakeProcess.instances[2].stopped
        calls.append(command)
    monkeypatch.setattr(workflow,'run_child',child)
    workflow.run_experiment(args,folder,saved_map)
    new=folder/'session'
    manifest=json.loads((new/'manifest.json').read_text())
    assert len(manifest['trials'])==1 and manifest['trials'][0]['id']==trial['id']
    assert not manifest.get('bidirectional') and 'retry_of' not in manifest
    assert (new/'paths/E2_environment.csv').read_bytes()==(session/'paths/E2_environment.csv').read_bytes()
    assert (folder/'map/map.yaml').read_bytes()==saved_map.read_bytes()
    assert len(workflow.prepared_inputs(tmp_path,new)[2])==1
    assert len(workflow.resume_inputs(tmp_path,new)[2])==1
    assert len(calls)==2 and '--dry-run' in calls[0]
    assert json.loads((folder/'reacquisition.json').read_text())['source_attempt_preserved']
    for name,data in before.items():
        assert Path(name).read_bytes()==data


def test_fixed_retry_rejects_modified_source_record(tmp_path):
    session,_,trial=interrupted_environment(tmp_path)
    file=session/'runs'/trial['id']/'trial.json'
    content=json.loads(file.read_text());content['manifest_sha256']='wrong'
    file.write_text(json.dumps(content))
    with pytest.raises(ValueError,match='frozen E2 inputs'):
        workflow.fixed_retry_inputs(tmp_path,session,trial['id'])


def test_early_background_exit_never_runs_next_child(tmp_path):
    marker = tmp_path/'must_not_run'
    with workflow.Process([sys.executable, '-c', 'raise SystemExit(7)'], tmp_path/'failed.log') as background:
        background.process.wait(timeout=3)
        with pytest.raises(RuntimeError, match='Process exited'):
            workflow.run_child([sys.executable, '-c', f'from pathlib import Path; Path({str(marker)!r}).touch()'],
                               tmp_path/'next.log', [background])
    assert not marker.exists()


def test_owned_child_receives_interrupt_and_cleanup(tmp_path):
    ready, stopped = tmp_path/'ready', tmp_path/'stopped'
    code = ('import signal,time,pathlib\n'
            f'def stop(*args):\n pathlib.Path({str(stopped)!r}).touch()\n raise SystemExit(0)\n'
            'signal.signal(signal.SIGINT,stop)\n'
            f'pathlib.Path({str(ready)!r}).touch()\n'
            'while True: time.sleep(.02)\n')
    with workflow.Process([sys.executable, '-c', code], tmp_path/'child.log') as process:
        deadline = time.monotonic()+3
        while not ready.exists() and time.monotonic()<deadline: time.sleep(.01)
        assert ready.exists()
    assert stopped.exists() and process.process.returncode == 0
