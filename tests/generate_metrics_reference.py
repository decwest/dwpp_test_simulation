"""Generate eight trajectories with the read-only simulator's locked Python 3.11.

Outputs belong in /tmp. No experiment-tool metrics are imported here.
"""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from omnidirectional_dwvp.access_metrics import evaluate
from omnidirectional_dwvp.config import Config
from omnidirectional_dwvp.paths import orientation_ramp_path, straight_path
from omnidirectional_dwvp.simulation import simulate
import omnidirectional_dwvp.access_metrics as source


def main():
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    base = Config(lookahead_time=.75)
    for condition, methods, scale in (
            ('E1_lateral', ('dwpp','dwvp'), 1.),
            ('E1_orientation_nominal', ('vp','vp_scaled','dwvp'), 1.),
            ('E1_orientation_half', ('vp','vp_scaled','dwvp'), .5)):
        lateral = condition == 'E1_lateral'
        path = straight_path(2.5, .01) if lateral else orientation_ramp_path(.3, length=2.5, start=1., spacing=.01)
        initial = [0., .5 if lateral else 0., 0.]
        config = replace(base, ax=base.ax*scale, ay=base.ay*scale, aw=base.aw*scale)
        for method in methods:
            result = simulate(path, method, config, initial_pose=initial)
            metrics = evaluate(result, config, {} if lateral else {'transition_length':.3}, initial, 1.9, 1.)
            filename = condition+'_'+method+'.npz'
            np.savez_compressed(output/filename, **result.arrays)
            cases.append(dict(condition=condition, method=method, initial=initial,
                              config=asdict(config), trajectory=filename, metrics=metrics,
                              trajectory_sha256=hashlib.sha256((output/filename).read_bytes()).hexdigest()))
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in Path(source.__file__).parent.glob('*.py')}
    (output/'reference.json').write_text(json.dumps(dict(
        purpose='Synthetic simulator trajectories, not physical HSR trials',
        python=sys.version, numpy=np.__version__, source_sha256=sources,
        absolute_tolerance=1e-9, relative_tolerance=0., cases=cases),indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
