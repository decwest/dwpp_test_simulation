"""Failed-only attempts preserve source evidence, frozen geometry and direction."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import dwvp_access_batch as batch
import dwvp_access_experiment as experiment
import dwvp_access_workflow as workflow
from test_access_batch import endpoint_miss, write_success
from test_access_workflow import fake_operator, FakeProcess, make_map


def completed_source(workspace):
    parent = workspace/'results/dwvp_access/lab/run_original'
    saved_map = make_map(parent/'map')
    session = parent/'session'
    manifest = experiment.prepare(session, ROOT/'params/hsrb_dwvp_access_params.yaml',
        current_start=[1.,2.,.7], bidirectional=True, conditions=['E1_orientation_quarter'], repeats=[1,2])
    for i, trial in enumerate(manifest['trials']):
        folder = write_success(session, trial)
        if i in (1,3): endpoint_miss(folder, legacy=i == 1)
    (session/'.batch.lock').touch()
    folder = session/'batches/original'; folder.mkdir(parents=True)
    experiment.write_json(folder/'map_input.json', dict(yaml_sha256=experiment.digest(saved_map),
        image_sha256=experiment.digest(saved_map.with_suffix('.pgm'))))
    workflow.atomic_json(workspace/'results/dwvp_access/latest.json', dict(session=str(session.relative_to(workspace))))
    return session, saved_map, manifest


def snapshot(folder):
    return {str(p.relative_to(folder)): p.read_bytes() for p in folder.rglob('*') if p.is_file()}


def verified_success_fixture(source, trial):
    file = source/'runs'/trial['id']/'result.json'
    result = json.loads(file.read_text())
    result.update(action_status=4, settling=dict(verified=True))
    experiment.write_json(file, result)


def test_explicit_success_retry_has_one_leg_and_preserves_success_evidence(tmp_path):
    source, _, original = completed_source(tmp_path)
    trial = original['trials'][0]
    verified_success_fixture(source, trial)
    before = snapshot(source)
    output = tmp_path/'selected_retry'
    retry = batch.prepare_retry(source, output, trial_id=trial['id'])
    assert retry['trials'] == [trial]
    assert batch.selected_trials(output)[1] == [trial]
    assert retry['retry_of']['selection'] == 'explicit_trial'
    assert retry['retry_of']['trial_id'] == trial['id']
    evidence = json.loads((output/'retry_source_results.json').read_text())
    assert evidence[trial['id']]['result']['success'] is True
    assert not (output/'runs').exists() and snapshot(source) == before
    np.testing.assert_array_equal(experiment.load_trial(source, trial['id'])[2],
                                  experiment.load_trial(output, trial['id'])[2])
    with pytest.raises(FileExistsError): batch.prepare_retry(source, output, trial_id=trial['id'])
    # A successful source must not slip into the legacy failed-only policy.
    retry['retry_of'].pop('selection')
    experiment.write_json(output/'manifest.json', retry)
    with pytest.raises(ValueError, match='failed source'): batch.selected_trials(output)


@pytest.mark.parametrize('damage', ['unknown', 'pending', 'unstopped', 'selection', 'wrong_id'])
def test_explicit_retry_rejects_invalid_source_or_selection(tmp_path, damage):
    source, _, original = completed_source(tmp_path)
    trial = original['trials'][0]
    verified_success_fixture(source, trial)
    output = tmp_path/'selected_retry'
    trial_id = trial['id']
    if damage == 'unknown': trial_id = '../unknown'
    if damage == 'pending':
        import shutil
        shutil.rmtree(source/'runs'/original['trials'][-1]['id'])
    if damage == 'unstopped':
        file = source/'runs'/trial_id/'result.json'
        result = json.loads(file.read_text()); result['settling']['verified'] = False
        experiment.write_json(file, result)
    if damage in ('selection', 'wrong_id'):
        retry = batch.prepare_retry(source, output, trial_id=trial_id)
        retry['retry_of']['selection' if damage == 'selection' else 'trial_id'] = 'unrecognized'
        experiment.write_json(output/'manifest.json', retry)
        with pytest.raises(ValueError): batch.selected_trials(output)
    else:
        with pytest.raises(ValueError): batch.prepare_retry(source, output, trial_id=trial_id)
        assert not output.exists()


def test_failed_subset_keeps_two_reverse_legs_and_original_records(tmp_path):
    source, _, original = completed_source(tmp_path)
    before = snapshot(source)
    output = tmp_path/'retry'
    retry = batch.prepare_failed_retry(source, output)
    expected = [original['trials'][i] for i in (1,3)]
    assert retry['trials'] == expected
    assert [t['direction'] for t in expected] == ['reverse','reverse']
    assert batch.selected_trials(output)[1] == expected
    assert snapshot(source) == before
    assert not (output/'runs').exists()
    for trial in expected:
        np.testing.assert_array_equal(experiment.load_trial(source, trial['id'])[2],
                                      experiment.load_trial(output, trial['id'])[2])
        np.testing.assert_array_equal(experiment.start_pose(original, trial), experiment.start_pose(retry, trial))
        assert (output/trial['params_file']).read_bytes() == (source/trial['params_file']).read_bytes()
    assert retry['retry_of']['manifest_sha256'] == experiment.digest(source/'manifest.json')
    assert all(v['result']['success'] is False for v in json.loads((output/'retry_source_results.json').read_text()).values())
    with pytest.raises(FileExistsError): batch.prepare_failed_retry(source, output)


@pytest.mark.parametrize('damage', ['direction', 'order', 'conditions', 'snapshot', 'input'])
def test_retry_rejects_changed_source_or_frozen_inputs(tmp_path, damage):
    source, _, _ = completed_source(tmp_path)
    output = tmp_path/'retry'
    retry = batch.prepare_failed_retry(source, output)
    if damage == 'direction': retry['trials'][0]['direction'] = 'forward'
    if damage == 'order': retry['trials'].reverse()
    if damage == 'conditions': retry['yaw_tolerance_rad'] = 1.
    if damage == 'snapshot': (output/'retry_source_results.json').write_text('{}')
    if damage == 'input': (output/retry['trials'][0]['params_file']).write_text('changed')
    experiment.write_json(output/'manifest.json', retry)
    with pytest.raises(ValueError): batch.selected_trials(output)


def test_retries_can_resume_and_successes_are_not_retried_again(tmp_path):
    source, _, _ = completed_source(tmp_path)
    output = tmp_path/'retry'
    retry = batch.prepare_failed_retry(source, output)
    first, second = retry['trials']
    write_success(output, first)
    assert batch.selected_trials(output, resume=True)[1] == [second]
    with pytest.raises(ValueError, match='pending'):
        batch.failed_trials(output)
    endpoint_miss(write_success(output, second))
    again = batch.prepare_failed_retry(output, tmp_path/'retry_again')
    assert again['trials'] == [second] and second['direction'] == 'reverse'
    assert batch.selected_trials(tmp_path/'retry_again')[1] == [second]


@pytest.mark.parametrize('policy', [batch.LEGACY_RETRY_DIRECTION_POLICY,batch.CURRENT_POSE_RETRY_DIRECTION_POLICY])
def test_old_retry_manifest_can_resume_without_rewriting_its_inputs(tmp_path,policy):
    source,_,_=completed_source(tmp_path)
    output=tmp_path/'retry'
    manifest=batch.prepare_failed_retry(source,output)
    manifest['direction_policy']=policy
    experiment.write_json(output/'manifest.json',manifest)
    before=snapshot(output)
    assert batch.selected_trials(output,resume=True)[1]==manifest['trials']
    assert snapshot(output)==before


@pytest.mark.parametrize('damage', ['missing', 'aborted', 'unstopped', 'all_success'])
def test_retry_never_relabels_unfinished_or_unverified_trials(tmp_path, damage):
    source, _, original = completed_source(tmp_path)
    file = source/'runs'/original['trials'][1]['id']/'result.json'
    if damage == 'missing': file.unlink()
    elif damage == 'all_success':
        for f in (source/'runs').glob('*/result.json'):
            experiment.write_json(f, dict(success=True, status='succeeded'))
    else:
        result = json.loads(file.read_text())
        if damage == 'aborted': result['action_status'] = 6
        if damage == 'unstopped': result['settling']['verified'] = False
        experiment.write_json(file, result)
    output = tmp_path/'retry'
    with pytest.raises(ValueError): batch.prepare_failed_retry(source, output)
    assert not output.exists()


@pytest.mark.parametrize('explicit', [False, True])
def test_retry_workflow_uses_frozen_source_and_creates_separate_attempt(monkeypatch, tmp_path, explicit):
    fake_operator(monkeypatch)
    source, saved_map, original = completed_source(tmp_path)
    trial_id = original['trials'][0]['id'] if explicit else None
    if explicit: verified_success_fixture(source, original['trials'][0])
    before = snapshot(source)
    folder = workflow.new_directory(source.parent/'retries', 'retry')
    args = NS(workspace=tmp_path, no_rviz=False, no_joy=False, resume=None,
              retry_failed=None if explicit else source, retry_trial=trial_id,
              retry_source=source, params=source/original['trials'][1]['params_file'])
    calls = []
    def child(command, logfile, background, timeout=None):
        assert FakeProcess.instances[-1].stopped
        assert '--resume' not in command and 'prepare' not in command
        assert '--start-from-current' in command and '--continue-on-endpoint-failure' in command
        assert command[command.index('--session')+1] == str(folder/'session')
        calls.append(command)
    monkeypatch.setattr(workflow, 'run_child', child)
    workflow.run_experiment(args, folder, saved_map)
    assert len(calls) == 2 and '--dry-run' in calls[0]
    assert snapshot(source) == before
    assert (folder/'map/map.pgm').read_bytes() == saved_map.with_suffix('.pgm').read_bytes()
    scheduled = batch.selected_trials(folder/'session')[1]
    assert len(scheduled) == (1 if explicit else 2)
    if explicit: assert scheduled[0]['id'] == trial_id
    assert all(p.stopped for p in FakeProcess.instances)


def test_retry_cli_dry_run_has_no_ros_or_file_side_effects(monkeypatch, tmp_path, capsys):
    source, _, original = completed_source(tmp_path)
    before = snapshot(tmp_path)
    monkeypatch.setattr(workflow, 'require_idle', lambda *a: pytest.fail('No ROS in dry run'))
    base = ['workflow', '--workspace', str(tmp_path), 'experiment', '--retry-failed']
    monkeypatch.setattr(sys, 'argv', base+['--dry-run'])
    workflow.main()
    printed = capsys.readouterr().out
    assert 'Failed trials to retry: 2' in printed
    assert all(original['trials'][i]['id'] in printed for i in (1,3))
    assert snapshot(tmp_path) == before
    monkeypatch.setattr(sys, 'argv', base+[str(source),'--conditions','E1','--dry-run'])
    with pytest.raises(ValueError, match='frozen conditions'): workflow.main()


def test_retry_cli_all_success_is_noop(monkeypatch, tmp_path, capsys):
    source, _, _ = completed_source(tmp_path)
    for file in (source/'runs').glob('*/result.json'):
        experiment.write_json(file, dict(success=True, status='succeeded'))
    before = snapshot(tmp_path)
    monkeypatch.setattr(workflow, 'require_idle', lambda *a: pytest.fail('No ROS when no failures'))
    monkeypatch.setattr(sys, 'argv', ['workflow','--workspace',str(tmp_path),'experiment','--retry-failed'])
    workflow.main()
    assert 'Failed trials to retry: 0' in capsys.readouterr().out
    assert snapshot(tmp_path) == before


def test_explicit_retry_cli_selects_one_success_and_never_starts_ros_in_dry_run(monkeypatch, tmp_path, capsys):
    source, _, original = completed_source(tmp_path)
    trial = original['trials'][0]; verified_success_fixture(source, trial)
    before = snapshot(tmp_path)
    monkeypatch.setattr(workflow, 'require_idle', lambda *a: pytest.fail('No ROS in dry run'))
    base = ['workflow', '--workspace', str(tmp_path), 'experiment', '--retry-trial', trial['id']]
    monkeypatch.setattr(sys, 'argv', base+['--source-session',str(source),'--dry-run'])
    workflow.main()
    printed = capsys.readouterr().out
    assert 'Explicit trials to retry: 1' in printed and trial['id'] in printed
    assert all(t['id'] not in printed for t in original['trials'][1:])
    assert snapshot(tmp_path) == before
    for extra in (['--dry-run'], ['--source-session',str(source),'--retry-failed','--dry-run']):
        monkeypatch.setattr(sys, 'argv', base+extra)
        with pytest.raises(SystemExit): workflow.main()
    monkeypatch.setattr(sys, 'argv', base+['--source-session',str(source),'--conditions','E1','--dry-run'])
    with pytest.raises(ValueError, match='frozen conditions'): workflow.main()
    assert snapshot(tmp_path) == before
