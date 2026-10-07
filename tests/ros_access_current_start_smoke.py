"""Exercise per-trial current-pose placement, only in Docker --network none."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import rclpy
import yaml

from ros_access_controller_smoke import Plant, spin_for
from ros_access_end_to_end_smoke import run_while_spinning

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import dwvp_access_experiment as experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert Path('/.dockerenv').exists() and set(os.listdir('/sys/class/net')) == {'lo'}
    assert os.environ.get('ROS_LOCALHOST_ONLY') == '1'
    out = args.output; out.mkdir(parents=True, exist_ok=False)
    config = experiment.default_config()
    for name, condition in config['conditions'].items():
        condition['methods'] = ['DWVP']
        if name != 'E2_environment':
            condition['length_m'] = .8
            condition['evaluation']['goal_margin_m'] = .1
        if name.startswith('E1_orientation'):
            condition.update(orientation_start_m=.2, orientation_length_m=.15)
    config['conditions']['E1_lateral']['start_pose'] = [0., .2, 0.]
    settings = out/'config.yaml'; settings.write_text(yaml.safe_dump(config))
    Image.fromarray(np.full((200,200),254,dtype=np.uint8)).save(out/'map.pgm')
    map_file = out/'map.yaml'
    map_file.write_text(yaml.safe_dump(dict(image='map.pgm', resolution=.05, origin=[-5.,-5.,0.],
                                          mode='trinary', negate=0, occupied_thresh=.65, free_thresh=.196)))
    session = out/'session'
    manifest = experiment.prepare(session, ROOT/'params/hsrb_dwvp_access_params.yaml',
        current_start=[0,0,0], config_file=settings, bidirectional=True, conditions=['E1'], repeats=[1])
    frozen = {str(p.relative_to(session)): p.read_bytes() for p in session.rglob('*') if p.is_file()}
    command = [sys.executable, str(ROOT/'scripts/dwvp_access_batch.py'), '--session', str(session),
               '--map', str(map_file), '--start-from-current']
    rclpy.init(); plant = Plant()
    try:
        # Deliberately outside the old start tolerance: first align toward the
        # frozen start, then use the measured residual instead of the target.
        plant.reset([.16, -.07, 0.]); spin_for(plant, 1.)
        before = plant.raw_count
        run_while_spinning(plant, command+['--dry-run'], out/'dry_run.log', 20.)
        assert plant.raw_count == before and not (session/'runs').exists()
        run_while_spinning(plant, command, out/'batch.log', 400.)
        for name, contents in frozen.items():
            assert (session/name).read_bytes() == contents
        captured_starts = []
        for trial in manifest['trials']:
            folder = session/'runs'/trial['id']
            initial, reference, policy = experiment.recorded_geometry(session, trial, folder)
            captured = json.loads((folder/'start_capture.json').read_text())['capture']['pose']
            np.testing.assert_allclose(initial, captured)
            assert policy == 'per_trial_current_pose'
            result = json.loads((folder/'result.json').read_text())
            assert result['success'] and result['settling']['verified']
            assert result['goal_checker_id'] == 'general_goal_checker'
            assert result['endpoint_policy'] == 'fixed_tolerance'
            packet = json.loads((folder/'start_capture.json').read_text())
            np.testing.assert_allclose(packet['alignment_target'], experiment.start_pose(manifest, trial))
            np.testing.assert_allclose(experiment.arclength(reference[:,:2])[-1], .8, atol=1e-8)
            captured_starts.append(initial.tolist())
        first_residual = np.linalg.norm(captured_starts[0][:2])
        assert .001 < first_residual < np.linalg.norm([.16,-.07])
        packets = list((session/'batches').glob('*/*_return.json'))
        assert len(packets) == 4
        for packet_file in packets:
            packet = json.loads(packet_file.read_text())
            path = np.array(packet['path'])
            trial_id = Path(packet['output']).name.removeprefix('start_')
            trial = next(t for t in manifest['trials'] if t['id']==trial_id)
            np.testing.assert_allclose(path[-1], experiment.start_pose(manifest, trial), atol=1e-6, rtol=0)
            result = json.loads((Path(packet['output'])/'result.json').read_text())
            assert packet['endpoint_policy'] == result['endpoint_policy'] == 'reanchor_after_stop'
            assert result['success'] and result['settling']['verified']
        report = experiment.summarize(session)
        assert report['recorded']==4 and all(t['success'] for t in report['trials'])
        lateral = next(t for t in report['trials'] if t['task']=='E1_lateral')
        np.testing.assert_allclose(lateral['lateral_initial_error_m'], .2, atol=1e-8)
        assert all(t['reference_policy']=='per_trial_current_pose' for t in report['trials'])
        before = plant.raw_count
        run_while_spinning(plant, command+['--resume'], out/'completed_resume.log', 20.)
        assert plant.raw_count == before
        (out/'report.json').write_text(json.dumps(dict(physical_trials=0, scored_trials=4,
            starts=captured_starts, alignments_to_frozen_starts=4, actual_geometry_metrics=True,
            frozen_inputs_unchanged=True, completed_resume_no_goals=True), indent=2)+'\n')
        print('PASS: four E1 conditions align toward fixed starts then reanchor to measured stops; recorded geometry, metrics and resume verified', flush=True)
    finally:
        plant.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()
