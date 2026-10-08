"""Expected recorder outcomes shared by the isolated integration smokes."""
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import dwvp_access_batch as batch
import dwvp_access_experiment as experiment


def assert_recorded_outcome(session, manifest, trial):
    folder = Path(session)/'runs'/trial['id']
    result = json.loads((folder/'result.json').read_text())
    assert result['action_status'] == 4, result  # Nav2 STATUS_SUCCEEDED
    assert result['settling']['verified'] is True, result
    assert result['final_pose_fresh'] is True, result
    assert all(math.isfinite(result[key]) and result[key] >= 0 for key in
               ('final_position_error_m', 'final_yaw_error_rad')), result
    if result['success']:
        assert batch.verified_success(result), result
        assert result['final_pose_within_tolerances'] is True, result
    else:
        # Only quarter-acceleration clipped VP may miss the stopped endpoint:
        # the smoother keeps decelerating after Nav2 succeeds. Three endpoint
        # failures were retained/retried on physical HSR on 2026-10-07; see
        # DWVP_ACCESS/data/hsr_lab_20261007/attempt_index.csv.
        assert (trial['task'], trial['controller']) == ('E1_orientation_quarter', 'VP_CLIP'), result
        assert result['failure_reason'] == 'endpoint_tolerance_exceeded', result
        assert batch.settled_endpoint_failure(result, manifest, folder), result
    return result


def assert_retry_bookkeeping(session, output):
    """Exercise the author's failed-only selection without replaying any trial."""
    session, output = Path(session), Path(output)
    manifest, failures = batch.failed_trials(session)
    for trial in manifest['trials']:
        assert_recorded_outcome(session, manifest, trial)
    before = {p: experiment.digest(p) for p in session.rglob('*') if p.is_file()}
    if failures:
        retry = batch.prepare_failed_retry(session, output)
        assert retry['trials'] == failures, retry
        assert batch.selected_trials(output)[1] == failures
        saved = json.loads((output/'retry_source_results.json').read_text())
        assert set(saved) == {t['id'] for t in failures}, saved
        for trial in failures:
            source = session/'runs'/trial['id']/'result.json'
            assert saved[trial['id']]['result_sha256'] == experiment.digest(source)
            assert saved[trial['id']]['result'] == json.loads(source.read_text())
        assert not (output/'runs').exists()
    assert all(experiment.digest(p) == digest for p, digest in before.items())
    return dict(failed_trial_ids=[t['id'] for t in failures],
                retry_trial_ids=[t['id'] for t in failures],
                retry_prepared=bool(failures), source_records_preserved=True)
