"""Known responses and acquisition boundaries; these are not hardware evidence."""
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from dwvp_actuation_metrics import step_metrics, summarize
from dwvp_actuation_step import main, parser, schedule
import dwvp_actuation_metrics as metrics
import dwvp_actuation_step as step


@pytest.mark.parametrize('sign', [1, -1])
def test_known_ramp_onset_rise_acceleration_and_braking(sign):
    t = np.arange(-1., 2.01, .005)
    v = np.where(t < 1., np.clip(.5*(t-.03), 0., .2), np.maximum(.2-.8*(t-1.), 0.))
    r = step_metrics(t, sign*v, sign*.2, 1., axis='x')
    # Onset threshold is 5% of plateau, not the pure 30 ms dead time.
    assert r['onset_lower_s'] <= .05 <= r['onset_upper_s']+1e-10
    assert r['rise_10_90_s'] == pytest.approx(.32, abs=.005)
    assert r['steady_velocity'] == pytest.approx(sign*.2)
    assert r['acceleration_peak'] == pytest.approx(.5)
    assert r['deceleration_peak'] == pytest.approx(.8)
    assert r['steady_reached'] and r['onset_resolved']
    assert not r['quality_flags']


def test_7ms_step_at_50hz_does_not_claim_resolved_7ms_lag():
    t = np.arange(-50, 101)*.02
    v = np.where((t >= .007) & (t < 1.), .2, 0.)
    r = step_metrics(t, v, .2, 1., axis='x')
    assert r['onset_lower_s'] == 0
    assert r['onset_upper_s'] == .02
    assert not r['onset_resolved']
    assert r['timing_resolution_s'] == pytest.approx(.02)
    bounded = step_metrics(t, v, .2, 1., axis='x', uncertainty=.005, publish_span=.001)
    assert bounded['onset_lower_s'] == pytest.approx(-.006)
    assert bounded['onset_upper_s'] == pytest.approx(.025)


def test_invalid_timestamps_gaps_and_noise_are_visible():
    t = np.arange(-100, 200)*.01
    v = np.where((t >= .1) & (t < 1.), .2, 0.)
    broken = t.copy(); broken[110] = broken[109]
    assert 'nonincreasing_timestamps' in step_metrics(broken, v, .2, 1., axis='yaw')['quality_flags']
    mask = (t < .05) | (t > .3)
    r = step_metrics(t[mask], v[mask], .2, 1., axis='x')
    assert 'sampling_gap' in r['quality_flags'] and not r['onset_resolved']
    r = step_metrics(t, np.zeros_like(t), .2, 1., axis='x')
    assert 'response_below_noise_floor' in r['quality_flags'] and r['onset_upper_s'] is None


def test_schedule_both_signs_and_explicit_bounds():
    args = parser().parse_args([])
    trials = schedule(args)
    assert len(trials) == 54 and len({t['trial_id'] for t in trials}) == 54
    for a, b in zip(trials[::2], trials[1::2]):
        assert (a['axis'], a['repeat'], a['amplitude']) == (b['axis'], b['repeat'], -b['amplitude'])
    for options in (['--linear-speeds', '.5'], ['--hold', '100'], ['--angular-limit', 'nan'], ['--clock-uncertainty-ms', '-1']):
        with pytest.raises(ValueError):
            schedule(parser().parse_args(options))


def test_dry_run_has_no_ros_or_files(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'rclpy', None)
    main(['--workspace', str(tmp_path), '--dry-run'])
    assert list(tmp_path.iterdir()) == []


@pytest.fixture
def recording(tmp_path):
    def write(name, rows):
        with (tmp_path/name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    (tmp_path/'session.json').write_text(json.dumps(dict(status='interrupted', settings={})))
    pulse = dict(trial_id='x_pos', axis='x', amplitude=.2, repeat=1, status='completed',
                 window_start_mono_s=99., window_end_mono_s=102., on_mono_s=100.,
                 on_publish_end_mono_s=100.0001, on_ros_s=1000., off_mono_s=101., off_ros_s=1001.)
    write('pulses.csv', [pulse, dict(pulse, trial_id='aborted', status='interrupted')])
    times = np.arange(-100, 201)*.01
    write('odom.csv', [dict(receive_mono_s=100.+t, receive_ros_s=1000.+t, source_ros_s=999.998+t,
                           vx=.2 if .08 <= t < 1. else 0., vy=0., omega=0.) for t in times])
    return tmp_path


def test_offline_analysis_keeps_clock_bases_and_incomplete_pulses_separate(recording):
    tmp_path = recording
    before = (tmp_path/'odom.csv').read_bytes()
    report = summarize(tmp_path)
    assert {r['time_basis'] for r in report['metrics']} == {'receive', 'none'}
    report = summarize(tmp_path, clock_uncertainty_ms=1.)
    assert {r['time_basis'] for r in report['metrics']} == {'receive', 'source', 'none'}
    assert report['sampling']['source_period_median_s'] == pytest.approx(.01)
    assert len(report['repeat_groups']) == 2
    assert all(g['recorded'] == 1 and g['steady_gain_sd'] is None for g in report['repeat_groups'])
    assert (tmp_path/'odom.csv').read_bytes() == before
    assert (tmp_path/'analysis/response_x.pdf').is_file()
    assert report['metrics'][-1]['quality_flags'] == ['incomplete_pulse']


def test_plotting_failure_prevents_robot_handoff_and_recording(tmp_path, monkeypatch):
    def broken():
        raise ImportError('numpy.core.multiarray failed to import')
    monkeypatch.setattr(step, 'check_plotting', broken)
    monkeypatch.setitem(sys.modules, 'dwvp_access_workflow', None)
    monkeypatch.setitem(sys.modules, 'rclpy', None)
    with pytest.raises(RuntimeError, match='Measurement was not started'):
        main(['--workspace', str(tmp_path), '--yes'])
    assert list(tmp_path.iterdir()) == []


def test_plotting_failure_preserves_metrics_and_allows_offline_recovery(recording, monkeypatch):
    before = {p.name: p.read_bytes() for p in recording.iterdir() if p.is_file()}
    monkeypatch.setitem(sys.modules, 'rclpy', None)
    def broken(*args):
        raise ImportError('numpy.core.multiarray failed to import')
    with monkeypatch.context() as patch:
        patch.setattr(metrics, 'plot_traces', broken)
        with pytest.raises(RuntimeError, match='Retry analysis without robot motion:'):
            main(['--analyze', str(recording)])
    for name in ('metrics.csv', 'group_metrics.csv', 'summary.json', 'report.md'):
        assert (recording/'analysis'/name).is_file()
    main(['--analyze', str(recording)])
    assert (recording/'analysis/response_x.pdf').is_file()
    assert (recording/'analysis/response_x.png').is_file()
    assert all((recording/name).read_bytes() == value for name, value in before.items())
