"""The integration allowance must reject other failures and preserve retries."""
import json

import pytest

from access_smoke_outcomes import (
    ROOT, assert_recorded_outcome, assert_retry_bookkeeping, experiment)
from test_access_batch import endpoint_miss, write_success


@pytest.fixture(params=['E1_orientation_half', 'E1_orientation_quarter'])
def completed_session(tmp_path, request):
    session = tmp_path/'session'
    manifest = experiment.prepare(session, ROOT/'params/hsrb_dwvp_access_params.yaml',
        current_start=[0, 0, 0], conditions=[request.param], repeats=[1])
    for trial in manifest['trials']:
        folder = write_success(session, trial)
        result = dict(status='succeeded', success=True, action_status=4,
            settling=dict(verified=True), final_pose_fresh=True,
            final_pose_within_tolerances=True, final_position_error_m=.05, final_yaw_error_rad=.1)
        experiment.write_json(folder/'result.json', result)
    trial = next(t for t in manifest['trials'] if t['controller'] == 'VP_CLIP')
    return session, manifest, trial


@pytest.mark.parametrize('failed', [False, True])
def test_outcome_and_failed_only_retry(completed_session, tmp_path, failed):
    session, manifest, trial = completed_session
    if failed:
        endpoint_miss(session/'runs'/trial['id'])
    assert assert_recorded_outcome(session, manifest, trial)['success'] is (not failed)
    report = assert_retry_bookkeeping(session, tmp_path/'retry')
    assert report['failed_trial_ids'] == report['retry_trial_ids'] == ([trial['id']] if failed else [])
    assert report['retry_prepared'] is failed
    assert report['source_records_preserved']
    assert json.loads((session/'runs'/trial['id']/'result.json').read_text())['success'] is (not failed)


@pytest.mark.parametrize('damage', [
    dict(action_status=6), dict(status='timeout'), dict(settling=dict(verified=False)),
    dict(final_pose_fresh=False), dict(final_position_error_m=float('nan')),
    dict(final_yaw_error_rad=float('inf')), dict(final_yaw_error_rad=-.4),
    dict(final_pose=[float('nan'), 0, 0]), dict(final_pose_within_tolerances=True),
    dict(final_position_error_m=.05, final_yaw_error_rad=.1),
    dict(failure_reason='controller_aborted'), dict(failure_reason=None),
    dict(error='unrelated failure'), dict(endpoint_policy='reanchor_after_stop'),
])
def test_expected_case_rejects_invalid_failure(completed_session, damage):
    session, manifest, trial = completed_session
    folder = session/'runs'/trial['id']
    result = endpoint_miss(folder)
    # Bypass the production writer to inject corrupt nonfinite recorded values.
    (folder/'result.json').write_text(json.dumps(dict(result, **damage)))
    with pytest.raises(AssertionError):
        assert_recorded_outcome(session, manifest, trial)


@pytest.mark.parametrize('task,controller', [
    ('E1_lateral', 'DWVP'), ('E1_lateral', 'DWPP'),
    ('E1_orientation_nominal', 'VP_CLIP'),
    ('E1_orientation_nominal', 'VP_SCALED'), ('E1_orientation_nominal', 'DWVP'),
    ('E1_orientation_half', 'VP_SCALED'), ('E1_orientation_half', 'DWVP'),
    ('E1_orientation_quarter', 'VP_SCALED'), ('E1_orientation_quarter', 'DWVP'),
    ('E2_environment', 'DWVP'), ('E2_environment', 'DWPP'),
    ('E2_environment', 'RPP'), ('E2_environment', 'MPPI'), ('E2_environment', 'DWB'),
    ('E2_environment', 'VP_CLIP'),
])
def test_other_cases_still_require_success(completed_session, task, controller):
    session, manifest, trial = completed_session
    endpoint_miss(session/'runs'/trial['id'])
    with pytest.raises(AssertionError):
        assert_recorded_outcome(session, manifest, dict(trial, task=task, controller=controller))
