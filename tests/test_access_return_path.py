"""Return through the checked E2 corridor, including resumes and blocked joins."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import dwvp_access_batch as batch
from dwvp_access_path import PathBlockedError, assert_clear


@pytest.fixture
def corridor():
    info = dict(resolution=.1, origin=[-1., -1., 0.])
    blocked = np.zeros((40, 40), dtype=bool)
    blocked[7:17, 16:24] = True  # obstacle between S=(0,0) and G=(2,0)
    path = np.array([[0.,0.,0.], [0.,1.2,0.], [2.,1.2,0.], [2.,0.,0.]])
    return path, (info, None, blocked)


@pytest.mark.parametrize('current', [[2.,0.,.3], [2.02,.5,.8], [.02,.15,1.]])
def test_returns_and_partial_resume_keep_the_corridor(corridor, tmp_path, current):
    path, data = corridor
    frozen = path.copy()
    target = [0.,0.,1.57]
    def unused(*args):
        pytest.fail('NavFn should not be used for a clear frozen corridor')
    route = batch.checked_return_path(current, target, data, .22, unused, tmp_path/'check.json', reference=path)
    assert np.allclose(route[0], current) and np.allclose(route[-1], target)
    assert np.array_equal(path, frozen)
    assert_clear(route, data[0], data[2], .22)
    report = json.loads((tmp_path/'check.json').read_text())
    assert report['verified'] and report['selected'] == 'frozen_reference_reverse'
    if current[0] > 1.:
        with pytest.raises(PathBlockedError):
            assert_clear([current,target], data[0], data[2], .22)
        assert [0.,1.2] in route[:,:2].tolist() and [2.,1.2] in route[:,:2].tolist()
    else:
        assert route[:,1].max() < .2  # do not revisit G after a partial return


def test_checked_alternative_if_reference_is_obstructed(corridor, tmp_path):
    path, data = corridor
    blocked = data[2].copy()
    blocked[20:24, 19:21] = True  # frozen top segment is obstructed
    fallback = np.array([[2.,0.,0.], [2.,-0.7,0.], [0.,-.7,0.], [0.,0.,0.]])
    route = batch.checked_return_path(path[-1], path[0], (data[0],None,blocked), .22,
        lambda *args: fallback, tmp_path/'check.json', reference=path)
    assert np.array_equal(route,fallback)
    report=json.loads((tmp_path/'check.json').read_text())
    assert report['selected']=='navfn' and report['candidates'][0]['obstruction']


def test_unsafe_routes_are_saved_and_never_accepted(corridor, tmp_path):
    path, data = corridor
    current=[1.,0.,0.]  # inside the obstacle
    record=tmp_path/'check.json'
    with pytest.raises(RuntimeError,match='no transfer goal sent'):
        batch.checked_return_path(current,path[0],data,.22,
            lambda *args: np.array([current,path[0]]),record,reference=path)
    report=json.loads(record.read_text())
    assert not report['verified'] and len(report['candidates'])==2
    assert all(c['obstruction'] for c in report['candidates'])
    assert report['candidates'][1]['path'][0]==current


def test_existing_planner_only_mode_remains_checked(corridor,tmp_path):
    path,data=corridor
    route=batch.checked_return_path(path[0],path[-1],data,.22,lambda *args:path,tmp_path/'check.json')
    assert np.array_equal(route,path)
    report=json.loads((tmp_path/'check.json').read_text())
    assert report['selected']=='navfn' and len(report['candidates'])==1
