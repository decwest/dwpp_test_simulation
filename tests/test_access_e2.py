"""Selection is based on completed common-metric integrals, never travel time."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from dwvp_access_e2 import grid, select


def row(name,p,h,success=True):
    return dict(controller='MPPI',candidate=name,overrides={},success=success,
                eval_position_error_integral_m_s=p,eval_heading_error_integral_deg_s=h)


def test_selection_normalizes_each_integral_and_excludes_failures():
    rows=[row('position',1,10),row('balanced',2,2),row('orientation',10,1),row('failed',.01,.01,False)]
    selected=select(rows)
    assert selected['MPPI']['candidate']=='balanced'
    assert selected['MPPI']['score']==4
    assert selected['DWB'] is None


def test_zero_minimum_and_tie_are_defined():
    rows=[row('first',0,2),row('second',0,2),row('positive',1,1)]
    assert select(rows)['MPPI']['candidate']=='first'
    assert rows[-1]['selection_score'] is None


def test_grid_is_exact():
    candidates=grid()
    assert len(candidates)==9
    assert len([c for c in candidates if c[0]=='DWB'])==6
    assert [o['PathAngleCritic']['cost_weight'] for c,_,o in candidates if c=='MPPI']==[6.,12.,20.]
