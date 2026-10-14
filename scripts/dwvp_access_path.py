#!/usr/bin/env python3
"""Generate an E2 NavFn path without hardware, in Docker --network none only.

Save map-coordinate x,y,yaw CSV, preview, planner settings and input hashes.
Position smoothing pins endpoints; optional fixed facing replaces tangent yaw.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import yaml


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def concatenate_segments(segments):
    """Join native NavFn positions in order, dropping coincident junctions only."""
    joined = []
    for segment in segments:
        points = np.asarray(segment, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
            raise ValueError('Each NavFn segment needs at least two finite XY points')
        if joined and np.linalg.norm(joined[-1]-points[0]) <= 1e-9:
            points = points[1:]
        joined.extend(points)
    if not joined:
        raise ValueError('At least one NavFn segment is required')
    return np.asarray(joined)


def facing_profile(path, facing, transition):
    """Apply fixed facing on the finite line corridor and wrapped smoothstep ramps."""
    path = np.asarray(path, dtype=float).copy()
    if not np.isfinite([*facing, transition]).all() or transition <= 0:
        raise ValueError('Finite facing and positive transition required')
    a, b = np.asarray(facing[:2]), np.asarray(facing[2:4])
    delta = b-a
    if np.dot(delta, delta) <= 1e-12:
        raise ValueError('Facing segment must have nonzero length')
    t = (path[:, :2]-a)@delta/np.dot(delta, delta)
    distance = np.linalg.norm(path[:, :2]-(a+t[:, None]*delta), axis=1)
    mask = (t >= 0) & (t <= 1) & (distance < .6)
    indices = np.flatnonzero(mask)
    if not len(indices):
        raise ValueError('No path points in the facing segment corridor')
    runs = np.split(indices, np.flatnonzero(np.diff(indices) > 1)+1)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1))]
    weights = np.zeros(len(path))
    ranges = []
    previous_end = -math.inf
    for run in runs:
        start, end = arc[run[0]], arc[run[-1]]
        if start-transition < 0 or end+transition > arc[-1] or start-transition < previous_end:
            raise ValueError('Facing transitions overlap or extend outside the path')
        ramp = np.minimum(np.clip((arc-(start-transition))/transition, 0, 1),
                          np.clip(((end+transition)-arc)/transition, 0, 1))
        weights = np.maximum(weights, ramp*ramp*(3-2*ramp))
        ranges.append(dict(first_index=int(run[0]), last_index=int(run[-1]),
            start_m=float(start), end_m=float(end), transition_start_m=float(start-transition),
            transition_end_m=float(end+transition)))
        previous_end = end+transition
    difference = np.arctan2(np.sin(facing[4]-path[:, 2]), np.cos(facing[4]-path[:, 2]))
    path[:, 2] += weights*difference
    path[:, 2] = np.arctan2(np.sin(path[:, 2]), np.cos(path[:, 2]))
    return path, dict(segment=[a.tolist(), b.tolist()], yaw_rad=float(facing[4]),
        transition_m=float(transition), corridor_distance_m=.6, intervals=ranges,
        selected_points=int(mask.sum()), interpolation='wrapped shortest-angle smoothstep in arc length')


def clearance_profile(path, info, occupied):
    """Exact point-to-occupied-cell-rectangle distance, not cell-centre distance."""
    resolution = float(info['resolution'])
    ox, oy, angle = info['origin']; c, s = math.cos(angle), math.sin(angle)
    points = (np.asarray(path)[:, :2]-[ox, oy])@np.array([[c, -s], [s, c]])
    cells = np.argwhere(occupied)[:, ::-1]
    if not len(cells):
        raise ValueError('Map has no occupied cells for clearance')
    centres = (cells+.5)*resolution
    distances, nearest = [], []
    for point in points:
        values = np.linalg.norm(np.maximum(np.abs(point-centres)-resolution/2, 0), axis=1)
        index = int(np.argmin(values))
        distances.append(values[index]); nearest.append(cells[index])
    return np.asarray(distances), np.asarray(nearest)


def remove_subcell_cusps(xy, resolution, deduplicate=False):
    """Remove NavFn's sub-cell reversals, which native Humble otherwise pins.

    A forward-only NavFn route has no intended reverse segments. Only an interior
    negative-dot-product vertex with BOTH adjacent edges <= one cell diagonal is
    removed. Endpoints and larger-scale bends are preserved. Log every deletion;
    the final circular-footprint collision check still applies.
    """
    points = [np.asarray(p,dtype=float) for p in xy]
    removed = []
    i = 1
    while i < len(points)-1:
        before, after = points[i]-points[i-1], points[i+1]-points[i]
        duplicate = deduplicate and min(np.linalg.norm(before), np.linalg.norm(after)) <= 1e-9
        if duplicate or (np.dot(before,after)<0 and max(np.linalg.norm(before),np.linalg.norm(after))<=math.sqrt(2)*resolution*(1+1e-6)):
            removed.append(points.pop(i).tolist())
            i=max(1,i-1)
        else:
            i+=1
    if deduplicate:
        # A via junction can contain A,B,A; removing its reversal B creates A,A.
        unique = [points[0]]
        for point in points[1:]:
            if np.linalg.norm(point-unique[-1]) <= 1e-9:
                removed.append(point.tolist())
            else:
                unique.append(point)
        points = unique
    return np.asarray(points), removed


def resample_positions(xy, spacing):
    """Resample AFTER Nav2 smoothing, preserving endpoints and at most spacing."""
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2 or not np.isfinite(xy).all():
        raise ValueError('At least two finite path positions are required')
    xy = xy[np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-9]]
    if len(xy) < 2:
        raise ValueError('Path has no length')
    if spacing <= 0 or not math.isfinite(spacing):
        raise ValueError('sample_spacing_m must be positive')
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    sample = np.linspace(0, arc[-1], max(2, int(math.ceil(arc[-1]/spacing))+1))
    return np.c_[np.interp(sample, arc, xy[:, 0]), np.interp(sample, arc, xy[:, 1])]


def tangent_path(xy, half_window_m=0.10):
    """Forward secant from s-0.10 to s+0.10 m, clipped at the endpoints."""
    xy = np.asarray(xy, dtype=float)
    if len(xy) < 2 or not np.isfinite(xy).all() or not math.isfinite(half_window_m) or half_window_m <= 0:
        raise ValueError('Finite positions and positive tangent window required')
    spacing = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    if np.any(spacing <= 1e-9):
        raise ValueError('Duplicate positions after smoothing')
    arc = np.r_[0., np.cumsum(spacing)]
    lo, hi = np.maximum(0., arc-half_window_m), np.minimum(arc[-1], arc+half_window_m)
    tangent = np.c_[np.interp(hi, arc, xy[:, 0])-np.interp(lo, arc, xy[:, 0]),
                    np.interp(hi, arc, xy[:, 1])-np.interp(lo, arc, xy[:, 1])]
    if np.any(np.linalg.norm(tangent, axis=1) <= 1e-9):
        raise ValueError('Undefined tangent (cusp) after smoothing')
    return np.c_[xy, np.arctan2(tangent[:, 1], tangent[:, 0])]


def rounded_graph(waypoints, radius=.3, spacing=.02):
    """Exact tangent line/circle fillets, sampled uniformly in analytic arc length.

    Trim distances are r*tan(abs(turn)/2). If two fillets consume an edge,
    shrink their radii proportionally; later edge corrections only shrink them.
    Exact reversals have no finite tangent fillet and are rejected explicitly.
    """
    points = np.asarray(waypoints, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or
            not np.isfinite(points).all() or not np.isfinite([radius, spacing]).all()
            or min(radius, spacing) <= 0):
        raise ValueError('Finite XY waypoints, positive radius and spacing required')
    edges = np.diff(points, axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths <= 1e-9):
        raise ValueError('Consecutive graph waypoints must be distinct')
    units = edges/lengths[:, None]
    turns = np.zeros(len(points))
    for i in range(1, len(points)-1):
        a, b = units[i-1], units[i]
        turns[i] = math.atan2(a[0]*b[1]-a[1]*b[0], np.dot(a, b))
    if np.any(np.abs(turns) >= math.pi-1e-8):
        raise ValueError('A reversing graph corner has no tangent circular fillet')
    trims = radius*np.tan(np.abs(turns)/2)
    for i, length in enumerate(lengths):
        total = trims[i]+trims[i+1]
        if total > length*(1-1e-9):
            trims[i:i+2] *= length*(1-1e-9)/total
    primitives, corners = [], []
    cursor = points[0]
    progress = 0.
    def line(end, direction):
        nonlocal cursor, progress
        length = float(np.linalg.norm(end-cursor))
        if length > 1e-10:
            primitives.append(dict(kind='line', start_m=progress, end_m=progress+length,
                                   start=cursor.tolist(), direction=direction.tolist()))
            progress += length
        cursor = end
    for i in range(1, len(points)-1):
        turn = turns[i]
        if abs(turn) < 1e-9:
            line(points[i], units[i-1])
            continue
        effective = trims[i]/math.tan(abs(turn)/2)
        entry = points[i]-trims[i]*units[i-1]
        exit_point = points[i]+trims[i]*units[i]
        line(entry, units[i-1])
        sign = math.copysign(1., turn)
        normal = sign*np.array([-units[i-1, 1], units[i-1, 0]])
        centre = entry+effective*normal
        start_angle = math.atan2(entry[1]-centre[1], entry[0]-centre[0])
        arc = dict(kind='arc', waypoint_index=i, requested_radius_m=float(radius),
                   radius_m=float(effective), radius_reduced=bool(effective < radius*(1-1e-8)),
                   centre=centre.tolist(), entry=entry.tolist(), exit=exit_point.tolist(),
                   start_angle_rad=start_angle, turn_rad=float(turn),
                   start_m=progress, end_m=progress+effective*abs(turn))
        primitives.append(arc); corners.append(arc.copy())
        progress = arc['end_m']; cursor = exit_point
    line(points[-1], units[-1])
    sample = np.linspace(0., progress, int(math.ceil(progress/spacing))+1)
    path = np.empty((len(sample), 3))
    for primitive in primitives:
        mask = (sample >= primitive['start_m']-1e-12) & (sample <= primitive['end_m']+1e-12)
        distance = np.clip(sample[mask]-primitive['start_m'], 0, primitive['end_m']-primitive['start_m'])
        if primitive['kind'] == 'line':
            direction = np.asarray(primitive['direction'])
            path[mask, :2] = primitive['start']+distance[:, None]*direction
            path[mask, 2] = math.atan2(direction[1], direction[0])
        else:
            sign = math.copysign(1., primitive['turn_rad'])
            theta = primitive['start_angle_rad']+sign*distance/primitive['radius_m']
            path[mask, :2] = primitive['centre']+primitive['radius_m']*np.c_[np.cos(theta), np.sin(theta)]
            path[mask, 2] = theta+sign*math.pi/2
    path[0, :2], path[-1, :2] = points[0], points[-1]
    path[:, 2] = np.arctan2(np.sin(path[:, 2]), np.cos(path[:, 2]))
    return path, dict(waypoints=points.tolist(), requested_corner_radius_m=float(radius),
                      corners=corners, primitives=primitives, analytic_length_m=progress,
                      radius_reductions=[c for c in corners if c['radius_reduced']],
                      sampling='uniform analytic arc length; exact line/circle tangent yaw')


def inflation_profile(path, cost_values, origin, resolution):
    """Sample published OccupancyGrid costs, retaining unknown/outside markers."""
    ox, oy, yaw = origin
    c, s = math.cos(yaw), math.sin(yaw)
    local = (np.asarray(path)[:, :2]-[ox, oy])@np.array([[c, -s], [s, c]])
    cells = np.floor(local/resolution).astype(int)
    h, w = cost_values.shape
    values = [int(cost_values[y, x]) if 0 <= x < w and 0 <= y < h else -2 for x, y in cells]
    return dict(max_published_cost=max(values), lethal_count=values.count(100),
                inscribed_count=values.count(99), unknown_or_outside_count=sum(v < 0 for v in values),
                zero_cost_count=values.count(0), samples=len(values),
                traversable=all(0 <= v < 99 for v in values), costs=values,
                cell_indices=cells.tolist())


def path_diagnostics(path, speed=0.22, omega_max=0.6):
    from dwvp_access_experiment import validate_path
    path = validate_path(path)
    if not np.isfinite([speed, omega_max]).all() or speed <= 0 or omega_max <= 0:
        raise ValueError('Positive finite speed and yaw-rate limit required')
    spacing = np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)
    arc = np.r_[0., np.cumsum(spacing)]
    yaw = np.unwrap(path[:, 2])
    # Required orientation rate, not an estimate from three noisy position points.
    curvature = np.gradient(yaw, arc)
    required = speed*curvature
    step = np.diff(yaw)
    excess = np.flatnonzero(np.abs(required) > omega_max)
    # Also expose geometric segment curvature so yaw filtering cannot hide position wiggles.
    segment_yaw = np.unwrap(np.arctan2(*np.diff(path[:, :2], axis=0).T[::-1]))
    geometric = np.diff(segment_yaw)/((spacing[:-1]+spacing[1:])/2)
    report = dict(points=len(path), length_m=float(arc[-1]), speed_m_s=speed, omega_max_rad_s=omega_max,
        curvature_definition='gradient(unwrapped CSV yaw, cumulative XY chord length)',
        spacing_m=dict(min=float(spacing.min()), mean=float(spacing.mean()), max=float(spacing.max())),
        max_abs_curvature_per_m=float(np.abs(curvature).max()),
        geometric_max_abs_curvature_per_m=float(np.abs(geometric).max()) if len(geometric) else 0.,
        required_yaw_rate_abs_p90_rad_s=float(np.percentile(np.abs(required), 90)),
        max_abs_required_yaw_rate_rad_s=float(np.abs(required).max()),
        largest_orientation_step_deg=float(np.degrees(np.abs(step).max())),
        largest_step_end_progress_m=float(arc[np.argmax(np.abs(step))+1]),
        orientation_steps_over_5_deg=int(np.sum(np.abs(step)>np.deg2rad(5))),
        excess_count=len(excess), excess_share_pct=100.*len(excess)/len(path),
        excess_points=[dict(index=int(i), progress_m=float(arc[i]), x=float(path[i,0]), y=float(path[i,1]),
                            curvature_per_m=float(curvature[i]), required_yaw_rate_rad_s=float(required[i])) for i in excess])
    profile = np.c_[arc, path, np.degrees(np.r_[0., step]), curvature, required]
    return report, profile, geometric


def facing_corner_gaps(path, graph, facing):
    """Measure gaps in CSV chord arc length, converting analytic corner bounds."""
    chord = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1))]
    analytic = np.linspace(0., graph['analytic_length_m'], len(path))
    corners = [dict(start_m=float(np.interp(c['start_m'], analytic, chord)),
                    end_m=float(np.interp(c['end_m'], analytic, chord))) for c in graph['corners']]
    result = []
    for region in facing['intervals']:
        previous = [c for c in corners if c['end_m'] <= region['start_m']]
        following = [c for c in corners if c['start_m'] >= region['end_m']]
        entry = region['transition_start_m']-max(c['end_m'] for c in previous) if previous else None
        exit_gap = min(c['start_m'] for c in following)-region['transition_end_m'] if following else None
        result.append(dict(entry_gap_m=entry, exit_gap_m=exit_gap,
            at_least_0p1_m=all(v is not None and v >= .1 for v in (entry, exit_gap))))
    return result


def check_path(csv_file, output, speed=0.22, omega_max=0.6, map_file=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path = np.loadtxt(csv_file, delimiter=',', skiprows=1)
    report, profile, geometric = path_diagnostics(path, speed, omega_max)
    tangent = tangent_path(path[:, :2])
    tangent_file = Path(csv_file).parent/'tangent_path.csv'
    if tangent_file.exists():
        tangent = np.loadtxt(tangent_file, delimiter=',', skiprows=1)
        if tangent.shape != path.shape or not np.allclose(tangent[:, :2], path[:, :2], atol=1e-12, rtol=0):
            raise ValueError('Saved analytic tangent does not match the inspected path')
    tangent_report, tangent_profile, _ = path_diagnostics(tangent, speed, omega_max)
    report['tangent_yaw'] = tangent_report
    maximum = report['geometric_max_abs_curvature_per_m']
    report['xy_min_radius_m'] = 1/maximum if maximum > 0 else None
    report['xy_radius_definition'] = 'inverse maximum absolute XY segment turning angle per mean adjacent chord length; null for straight path'
    metadata_file = Path(csv_file).parent/'path_metadata.json'
    if metadata_file.exists():
        metadata = json.loads(metadata_file.read_text())
        if metadata.get('graph'):
            if metadata.get('facing'):
                report['facing_corner_gaps'] = facing_corner_gaps(path, metadata['graph'], metadata['facing'])
            # Finite differences/secants cross boundaries by up to their window.
            arcs = metadata['graph']['corners']
            facing = (metadata.get('facing') or {}).get('intervals', [])
            for diagnostics, tolerance in ((report, report['spacing_m']['max']), (tangent_report, .1+report['spacing_m']['max'])):
                counts = dict(corner=0, facing_transition=0, outside_expected_intervals=0)
                for point in diagnostics['excess_points']:
                    s = point['progress_m']
                    if any(c['start_m']-tolerance <= s <= c['end_m']+tolerance for c in arcs):
                        counts['corner'] += 1
                    elif any(r['transition_start_m']-tolerance <= s <= r['start_m']+tolerance or
                             r['end_m']-tolerance <= s <= r['transition_end_m']+tolerance for r in facing):
                        counts['facing_transition'] += 1
                    else:
                        counts['outside_expected_intervals'] += 1
                diagnostics['excess_interval_counts'] = counts
    clearance = np.full(len(path), np.nan)
    if map_file is None and (Path(csv_file).parent/'map.yaml').exists():
        map_file = Path(csv_file).parent/'map.yaml'
    report['clearance'] = None
    inflation_file = Path(csv_file).parent/'inflation_check.json'
    report['inflation'] = json.loads(inflation_file.read_text()).get('path_samples') if inflation_file.exists() else None
    if map_file is not None:
        info, pixels, blocked = load_map(map_file)
        occupancy = pixels/255 if info.get('negate', 0) else (255-pixels)/255
        clearance, nearest = clearance_profile(path, info, np.flipud(occupancy > info['occupied_thresh']))
        index = int(np.argmin(clearance))
        blocked_clearance, _ = clearance_profile(path, info, blocked)
        report['clearance'] = dict(definition='distance from path centre to nearest occupied cell rectangle; map includes walls and table legs, not tabletop overhang',
            min_m=float(clearance[index]), index=index, progress_m=float(profile[index, 0]),
            position=path[index, :2].tolist(), nearest_occupied_cell=nearest[index].tolist(),
            min_occupied_or_unknown_m=float(blocked_clearance.min()), map_yaml_sha256=sha(map_file))
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    report.update(csv_sha256=sha(csv_file), source_csv=str(csv_file))
    np.savetxt(output/'profile.csv', np.c_[profile, tangent[:, 2], tangent_profile[:, 5:7], clearance], delimiter=',', comments='',
               header='progress_m,x,y,yaw,yaw_step_deg,curvature_per_m,required_yaw_rate_rad_s,tangent_yaw,tangent_curvature_per_m,tangent_required_yaw_rate_rad_s,clearance_m')
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    for ax, values, label in zip(axes, [np.degrees(np.unwrap(path[:,2])), clearance, profile[:,5], profile[:,6]],
                               ['yaw [deg]', 'clearance [m]', 'curvature [1/m]', 'required yaw rate [rad/s]']):
        ax.plot(profile[:,0], values); ax.set_ylabel(label); ax.grid(True, alpha=.3)
    axes[0].plot(profile[:,0], np.degrees(np.unwrap(tangent[:,2])), '--', label='tangent yaw')
    axes[0].lines[0].set_label('reference yaw profile')
    axes[0].legend()
    if map_file is None:
        axes[1].text(.1,.5,'No map supplied',transform=axes[1].transAxes)
    axes[2].plot(profile[1:-1,0], geometric, alpha=.4, label='XY segment curvature')
    axes[2].lines[0].set_label('reference yaw derivative')
    axes[2].legend()
    axes[3].plot(profile[:,0], tangent_profile[:,6], '--', label='tangent yaw')
    axes[3].lines[0].set_label('reference yaw profile')
    axes[3].legend()
    for limit in (-omega_max, omega_max):
        axes[3].axhline(limit, color='red', linestyle='--')
    axes[-1].set_xlabel('progress [m]')
    fig.tight_layout(); fig.savefig(output/'path_check.png', dpi=160); plt.close(fig)
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def load_map(file):
    from PIL import Image
    file = Path(file)
    info = yaml.safe_load(file.read_text())
    if info.get('mode','trinary') != 'trinary':
        raise ValueError('Map validation currently supports trinary occupancy maps only')
    pixels = np.asarray(Image.open(file.parent/info['image']).convert('L'))
    occupancy = pixels/255 if info.get('negate',0) else (255-pixels)/255
    # Unknown is blocked. This also rejects cells at either threshold boundary.
    blocked = occupancy >= info['free_thresh']
    return info,pixels,np.flipud(blocked)


class PathBlockedError(ValueError):
    def __init__(self, details):
        self.details = details
        x, y = details['path_point_map']
        super().__init__(f'Path footprint intersects occupied/unknown/outside map at '
                         f'({x:.3f}, {y:.3f}) m; check alignment and path endpoints')


def assert_clear(path, info, blocked, radius):
    resolution = float(info['resolution'])
    if resolution <= 0 or radius < 0:
        raise ValueError('Invalid map resolution or robot radius')
    xy = np.asarray(path)[:,:2]
    # Densify segments so smoothing cannot cut through an obstacle between samples.
    dense=[]
    for a,b in zip(xy[:-1],xy[1:]):
        dense.extend(a+np.linspace(0,1,max(2,int(math.ceil(np.linalg.norm(b-a)/(resolution/2)))+1))[:,None]*(b-a))
    ox,oy,angle = info['origin'];c,s=math.cos(angle),math.sin(angle)
    local=(np.asarray(dense)-[ox,oy])@np.array([[c,-s],[s,c]])
    reach=int(math.ceil(radius/resolution))+1
    height,width=blocked.shape
    for point, point_map in zip(local, dense):
        mx,my=np.floor(point/resolution).astype(int)
        for gy in range(my-reach,my+reach+1):
            for gx in range(mx-reach,mx+reach+1):
                centre=(np.array([gx,gy])+.5)*resolution
                # Exact distance from centre position to the cell rectangle.
                distance=np.linalg.norm(np.maximum(np.abs(point-centre)-resolution/2,0))
                if distance<=radius and (gx<0 or gy<0 or gx>=width or gy>=height or blocked[gy,gx]):
                    cell_map = centre@np.array([[c,s],[-s,c]])+[ox,oy]
                    raise PathBlockedError(dict(path_point_map=point_map.tolist(),
                        cell_map=cell_map.tolist(), cell_index=[int(gx),int(gy)],
                        outside_map=bool(gx<0 or gy<0 or gx>=width or gy>=height),
                        robot_radius_m=float(radius), distance_to_cell_m=float(distance)))


def plot_path(file, path, raw, info, pixels, graph=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.transforms import Affine2D
    fig,ax=plt.subplots(figsize=(8,7))
    h,w=pixels.shape;res=info['resolution'];ox,oy,yaw=info['origin']
    transform=Affine2D().rotate(yaw).translate(ox,oy)+ax.transData
    ax.imshow(pixels,cmap='gray',vmin=0,vmax=255,origin='upper',extent=[0,w*res,0,h*res],transform=transform)
    ax.plot(raw[:,0],raw[:,1],color='tab:orange',alpha=.7,label='Graph edges' if graph else 'NavFn positions')
    ax.plot(path[:,0],path[:,1],color='tab:blue',label='Rounded graph' if graph else 'Smoothed path')
    step=max(1,len(path)//30)
    ax.quiver(path[::step,0],path[::step,1],np.cos(path[::step,2]),np.sin(path[::step,2]),
              angles='xy',scale_units='xy',scale=6,color='tab:red',width=.004)
    ax.scatter(*path[0,:2],color='green',label='Path start');ax.scatter(*path[-1,:2],color='purple',label='Path end')
    ax.set_xlim(path[:,0].min()-.7,path[:,0].max()+.7);ax.set_ylim(path[:,1].min()-.7,path[:,1].max()+.7)
    ax.set_aspect('equal');ax.set_xlabel('map x [m]');ax.set_ylabel('map y [m]');ax.legend()
    fig.tight_layout();fig.savefig(file,dpi=160);plt.close(fig)


def planner_parameters(map_path, config, resolution):
    p=config['planner']
    return {'map_server':{'ros__parameters':{'yaml_filename':str(map_path)}},
            'planner_server':{'ros__parameters':{'expected_planner_frequency':1.0,'planner_plugins':['NavFn'],
                'NavFn':{'plugin':'nav2_navfn_planner/NavfnPlanner',**{k:p[k] for k in ('tolerance','use_astar','allow_unknown')}}}},
            'global_costmap':{'global_costmap':{'ros__parameters':{'update_frequency':1.0,'publish_frequency':1.0,
                'global_frame':'map','robot_base_frame':'base_link','robot_radius':p['robot_radius_m'],
                'resolution':resolution,'track_unknown_space':True,'plugins':['static_layer','inflation_layer'],
                'static_layer':{'plugin':'nav2_costmap_2d::StaticLayer','map_subscribe_transient_local':True},
                'inflation_layer':{'plugin':'nav2_costmap_2d::InflationLayer','inflation_radius':p['inflation_radius_m'],
                                  'cost_scaling_factor':p['cost_scaling_factor']},'always_send_full_costmap':True}}}}


def smoother_parameters(config):
    smoothing = config['smoothing']
    if smoothing['method'] == 'nav2_simple_smoother':
        name = 'simple_smoother'
        plugin = 'nav2_smoother::SimpleSmoother'
        options = {k: smoothing[k] for k in ('w_data', 'w_smooth', 'max_its', 'tolerance', 'do_refinement')}
        if 'refinement_num' in smoothing:
            options['refinement_num'] = smoothing['refinement_num']
    elif smoothing['method'] == 'nav2_constrained_smoother':
        name = 'constrained_smoother'
        plugin = 'nav2_constrained_smoother/ConstrainedSmoother'
        options = {'minimum_turning_radius': smoothing['minimum_turning_radius']}
    else:
        raise ValueError('Unsupported Nav2 smoother')
    return name, plugin, {'ros__parameters': {
        'smoother_plugins': [name], name: {'plugin': plugin, **options}}}


def generate(args):
    if not Path('/.dockerenv').exists() or set(os.listdir('/sys/class/net')) != {'lo'} or os.environ.get('ROS_LOCALHOST_ONLY') != '1':
        raise RuntimeError('Run in Docker --network none with ROS_LOCALHOST_ONLY=1; no hardware connection is allowed')
    import rclpy
    from ament_index_python.packages import get_package_prefix
    from action_msgs.msg import GoalStatus
    from geometry_msgs.msg import PoseStamped
    from nav2_msgs.action import ComputePathToPose, SmoothPath
    from nav_msgs.msg import OccupancyGrid
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile
    import shutil
    graph_mode = getattr(args, 'graph', False)
    via = getattr(args, 'via', None) or []
    if not np.isfinite([*args.start,*args.goal,*np.asarray(via).ravel()]).all():
        raise ValueError('Finite start, via and goal required')
    config=yaml.safe_load(args.config.read_text())
    info,pixels,blocked=load_map(args.map)
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(args.map,output/'input_map.yaml')
    image_name = 'input_map_image' + Path(info['image']).suffix
    shutil.copy2(args.map.parent/info['image'],output/image_name)
    local_map=dict(info,image=image_name)
    (output/'map.yaml').write_text(yaml.safe_dump(local_map))
    (output/'path_config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    params=planner_parameters(output/'map.yaml',config,info['resolution'])
    inspection_resolution = config.get('inspection_costmap_resolution_m', info['resolution']) if graph_mode else info['resolution']
    check_frame = 'base_link'
    check_position = args.start[:2]
    expected_width, expected_height = pixels.shape[1], pixels.shape[0]
    if graph_mode:
        params.pop('planner_server')
        params['global_costmap']['global_costmap']['ros__parameters']['static_layer']['map_topic'] = '/map'
        params['costmap'] = {'costmap': params.pop('global_costmap')['global_costmap']}
        if inspection_resolution != info['resolution']:
            if not math.isfinite(inspection_resolution) or inspection_resolution <= 0 or info['origin'][2] != 0:
                raise ValueError('Inspection resolution must be positive; custom inspection requires an unrotated map')
            # A nonrolling StaticLayer resizes to the source-map resolution.
            # Use a stationary rolling inspection window covering the entire map;
            # the native layer samples the original map at the requested resolution.
            width = math.ceil(pixels.shape[1]*info['resolution'])
            height = math.ceil(pixels.shape[0]*info['resolution'])
            check_frame = 'e2_path_check_centre'
            check_position = [info['origin'][0]+width/2, info['origin'][1]+height/2]
            values = params['costmap']['costmap']['ros__parameters']
            values.update(rolling_window=True, width=width, height=height,
                          resolution=float(inspection_resolution), robot_base_frame=check_frame,
                          origin_x=info['origin'][0], origin_y=info['origin'][1])
            expected_width, expected_height = int(width/inspection_resolution), int(height/inspection_resolution)
    else:
        smoother_name, smoother_plugin, params['smoother_server'] = smoother_parameters(config)
    paramfile=output/'planner_params.yaml';paramfile.write_text(yaml.safe_dump(params,sort_keys=False))
    processes=[];logs=[];rclpy.init();node=Node('dwvp_access_path_generator')
    costmaps=[]
    def receive_costmap(msg):
        costmaps[:] = [msg]
    costmap_sub=node.create_subscription(OccupancyGrid,'/costmap/costmap' if graph_mode else '/global_costmap/costmap',
        receive_costmap,
        QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
    def spawn(package,executable,extra):
        log=(output/(executable+'.log')).open('w');logs.append(log)
        # Own the node process itself so shutdown cannot leave ros2-run children alive.
        binary=Path(get_package_prefix(package))/'lib'/package/executable
        processes.append(subprocess.Popen([str(binary),*extra],stdout=log,stderr=subprocess.STDOUT))
    try:
        spawn('tf2_ros','static_transform_publisher',['--x',str(check_position[0]),'--y',str(check_position[1]),'--yaw',str(args.start[2]),'--frame-id','map','--child-frame-id',check_frame])
        servers = [('nav2_map_server','map_server')]
        if graph_mode:
            servers.append(('nav2_costmap_2d','nav2_costmap_2d'))
            lifecycle_names = ['map_server', '/costmap/costmap']
        else:
            servers += [('nav2_planner','planner_server'), ('nav2_smoother','smoother_server')]
            lifecycle_names = [executable for _, executable in servers]
        for package,executable in servers:
            spawn(package,executable,['--ros-args','--params-file',str(paramfile)])
        spawn('nav2_lifecycle_manager','lifecycle_manager',['--ros-args','-p','autostart:=true','-p',
              'node_names:='+str(lifecycle_names),'-p','bond_timeout:=0.0'])
        if not graph_mode:
            client=ActionClient(node,ComputePathToPose,'/compute_path_to_pose')
            if not client.wait_for_server(timeout_sec=60):
                raise RuntimeError('NavFn action unavailable; see saved node logs')
        # Action availability can precede the first static-layer update. Plan only
        # after a published, initialized costmap matches the map loaded above.
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            rclpy.spin_once(node,timeout_sec=.1)
            if (costmaps and costmaps[0].header.frame_id=='map'
                    and costmaps[0].info.width==expected_width
                    and costmaps[0].info.height==expected_height
                    and math.isclose(costmaps[0].info.resolution,inspection_resolution,rel_tol=1e-6)
                    and any(value>=0 for value in costmaps[0].data)):
                break
        else:
            raise RuntimeError('Initialized global costmap unavailable; see saved node logs')
        costmap = costmaps[0]
        cost_values = np.asarray(costmap.data).reshape(costmap.info.height, costmap.info.width)
        np.savetxt(output/'inflated_costmap.csv', cost_values, delimiter=',', fmt='%d')
        waypoints = [args.start, *[[*point, 0.] for point in via], args.goal]
        cells = []
        for point in waypoints:
            gx = int(math.floor((point[0]-costmap.info.origin.position.x)/costmap.info.resolution))
            gy = int(math.floor((point[1]-costmap.info.origin.position.y)/costmap.info.resolution))
            value = int(cost_values[gy, gx]) if 0 <= gx < costmap.info.width and 0 <= gy < costmap.info.height else None
            cells.append(dict(position=point[:2], cell_index=[gx, gy], published_cost=value))
        (output/'inflation_check.json').write_text(json.dumps(dict(waypoints=cells,
            published_cost_definition='OccupancyGrid: 100 lethal obstacle, 99 inscribed inflation, 1..98 nonlethal cost, -1 unknown',
            robot_diameter_m=2*config['planner']['robot_radius_m'],
            inflation_radius_m=config['planner']['inflation_radius_m'],
            note='Inflation radius is a soft cost extent, not a required clearance; dense footprint check also applies'), indent=2)+'\n')
        def pose(values):
            p=PoseStamped();p.header.frame_id='map';p.header.stamp=node.get_clock().now().to_msg()
            p.pose.position.x=float(values[0]);p.pose.position.y=float(values[1])
            p.pose.orientation.z=math.sin(values[2]/2);p.pose.orientation.w=math.cos(values[2]/2)
            return p
        graph_metadata = None
        if graph_mode:
            raw = np.asarray([p[:2] for p in waypoints])
            path, graph_metadata = rounded_graph(raw, args.corner_radius, config['smoothing']['sample_spacing_m'])
            removed = []; segments = []; actual = {}
            smoother_plugin = None; smoother_name = None
        else:
            segments = []
            for index, (start, end) in enumerate(zip(waypoints[:-1], waypoints[1:])):
                goal=ComputePathToPose.Goal();goal.start=pose(start);goal.goal=pose(end);goal.use_start=True;goal.planner_id='NavFn'
                future=client.send_goal_async(goal);rclpy.spin_until_future_complete(node,future,timeout_sec=30)
                if not future.done() or future.result() is None or not future.result().accepted:
                    raise RuntimeError(f'NavFn rejected segment {index} or timed out')
                handle=future.result();future=handle.get_result_async();rclpy.spin_until_future_complete(node,future,timeout_sec=60)
                if not future.done() or future.result().status!=GoalStatus.STATUS_SUCCEEDED:
                    raise RuntimeError(f'NavFn segment {index} ({start[:2]} -> {end[:2]}) failed or timed out')
                navfn_path = future.result().result.path
                segment = np.array([[p.pose.position.x,p.pose.position.y] for p in navfn_path.poses])
                segments.append(segment)
                np.savetxt(output/f'navfn_segment_{index}.csv',segment,delimiter=',',header='x,y',comments='')
            raw = concatenate_segments(segments)
            if via:
                from dwvp_access_experiment import pose_path_message
                navfn_path = pose_path_message(tangent_path(raw), 'map', node.get_clock().now().to_msg())
            removed = []
            if config['planner'].get('remove_subcell_cusps', False):
                cleaned, removed = remove_subcell_cusps(raw, info['resolution'], deduplicate=bool(via))
                # Use only retained native poses; do not perform custom position smoothing.
                from dwvp_access_experiment import pose_path_message
                navfn_path = pose_path_message(tangent_path(cleaned), 'map', node.get_clock().now().to_msg())
            if smoother_name == 'constrained_smoother':
                # NavFn snaps its first pose to a cell. Retain requested endpoints.
                navfn_path.poses[0].pose.position.x = args.start[0]
                navfn_path.poses[0].pose.position.y = args.start[1]
                # One-sided forward yaw prevents a centred tangent around a sharp
                # junction from labelling a forward segment as a reversing cusp.
                for current, following in zip(navfn_path.poses[:-1], navfn_path.poses[1:]):
                    yaw = math.atan2(following.pose.position.y-current.pose.position.y,
                                     following.pose.position.x-current.pose.position.x)
                    current.pose.orientation.z = math.sin(yaw/2)
                    current.pose.orientation.w = math.cos(yaw/2)
                # The native smoother preserves endpoint orientation by default.
                navfn_path.poses[0].pose.orientation = pose(args.start).pose.orientation
                navfn_path.poses[-1].pose.orientation = pose(args.goal).pose.orientation
            np.savetxt(output/'smoother_input_positions.csv',
                       [[p.pose.position.x,p.pose.position.y] for p in navfn_path.poses],
                       delimiter=',',header='x,y',comments='')
            smoother = ActionClient(node, SmoothPath, '/smooth_path')
            if not smoother.wait_for_server(timeout_sec=30):
                raise RuntimeError('Smoother action unavailable')
            # Action discovery precedes lifecycle activation; wait for ACTIVE explicitly.
            from lifecycle_msgs.srv import GetState
            from lifecycle_msgs.msg import State
            state_client = node.create_client(GetState, '/smoother_server/get_state')
            deadline = time.monotonic()+30
            while time.monotonic()<deadline:
                if state_client.wait_for_service(timeout_sec=.1):
                    state_future = state_client.call_async(GetState.Request())
                    rclpy.spin_until_future_complete(node,state_future,timeout_sec=1.)
                    if state_future.done() and state_future.result().current_state.id==State.PRIMARY_STATE_ACTIVE:
                        break
                rclpy.spin_once(node,timeout_sec=.1)
            else:
                raise RuntimeError('Smoother lifecycle did not activate')
            node.destroy_client(state_client)
            from dwvp_access_experiment import snapshot_parameters
            runtime = snapshot_parameters(node, 'smoother_server')
            (output/'smoother_server_runtime.yaml').write_text(yaml.safe_dump(runtime))
            actual = runtime['/smoother_server']['ros__parameters']
            if ('refinement_num' in config['smoothing'] and 'simple_smoother.refinement_num' not in actual):
                raise ValueError('This Nav2 does not support refinement_num; omit it to use the documented native behavior')
            request = SmoothPath.Goal(); request.path = navfn_path
            request.smoother_id = smoother_name; request.max_smoothing_duration.sec = 10
            request.check_for_collisions = True
            future = smoother.send_goal_async(request)
            rclpy.spin_until_future_complete(node, future, timeout_sec=30)
            if not future.done() or future.result() is None or not future.result().accepted:
                raise RuntimeError('Smoother rejected request or timed out')
            future = future.result().get_result_async()
            rclpy.spin_until_future_complete(node, future, timeout_sec=30)
            if not future.done() or future.result().status != GoalStatus.STATUS_SUCCEEDED or not future.result().result.was_completed:
                raise RuntimeError('Nav2 smoothing failed or was incomplete')
            smoothed = np.array([[p.pose.position.x,p.pose.position.y] for p in future.result().result.path.poses])
            np.savetxt(output/(smoother_name+'_positions.csv'),smoothed,delimiter=',',header='x,y',comments='')
            path=tangent_path(resample_positions(smoothed,config['smoothing']['sample_spacing_m']),
                              config['smoothing']['tangent_half_window_m'])
        facing = None
        if graph_mode:
            np.savetxt(output/'tangent_path.csv', path, delimiter=',', header='x,y,yaw', comments='')
        if getattr(args, 'facing', None) is not None:
            path, facing = facing_profile(path, args.facing, args.transition)
        assert_clear(path,info,blocked,config['planner']['robot_radius_m'])
        inflation_file = output/'inflation_check.json'
        inflation = json.loads(inflation_file.read_text())
        q = costmap.info.origin.orientation
        origin = [costmap.info.origin.position.x, costmap.info.origin.position.y,
                  math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))]
        inflation['path_samples'] = inflation_profile(path, cost_values, origin, costmap.info.resolution)
        inflation['path_samples']['resolution_m'] = float(costmap.info.resolution)
        inflation['path_samples']['origin'] = origin
        inflation_file.write_text(json.dumps(inflation, indent=2)+'\n')
        if not inflation['path_samples']['traversable']:
            raise ValueError('Path samples include lethal, inscribed, unknown or outside costmap cells')
        np.savetxt(output/('graph_waypoints.csv' if graph_mode else 'navfn_positions.csv'),raw,delimiter=',',header='x,y',comments='')
        np.savetxt(output/'E2_environment.csv',path,delimiter=',',header='x,y,yaw',comments='')
        plot_path(output/'preview.png',path,raw,info,pixels,graph=graph_mode)
        start = list(args.start)
        if getattr(args, 'align_start_yaw', False):
            start[2] = float(path[0,2])
        metadata={'frame':'map','planner':'rounded_route_graph' if graph_mode else 'NavFn', 'graph':graph_metadata,'start':start,'requested_start':args.start,'requested_goal':args.goal,
                  'start_yaw_minus_csv_yaw_rad':float(math.atan2(math.sin(start[2]-path[0,2]), math.cos(start[2]-path[0,2]))),
                  'via':via,'facing':facing,'navfn_segments':len(segments),
                  'junction_policy':'concatenate before cusp removal and one native smoothing action; junctions not pinned',
                  'terminal_yaw_policy':'forward tangent, including endpoint; requested goal yaw is not retained',
                  'csv_sha256':sha(output/'E2_environment.csv'),'map_yaml_sha256':sha(args.map),
                  'map_image_sha256':sha(args.map.parent/info['image']),'planner_params_sha256':sha(paramfile),
                  'path_config_sha256':sha(output/'path_config.yaml'),'smoothing':config['smoothing'],
                  'footprint_check':'circular footprint against occupied, unknown and outside-map cells',
                  'costmap_ready_before_goal':True,
                  'smoother':smoother_plugin, 'smoother_action_completed':not graph_mode,
                  'removed_subcell_cusps':removed,
                  'refinement_num':actual.get('simple_smoother.refinement_num', 4) if smoother_name == 'simple_smoother' else None,
                  'refinement_note':'Humble SimpleSmoother uses four hardcoded refinements' if smoother_name == 'simple_smoother' else 'ConstrainedSmoother native defaults except minimum_turning_radius',
                  'physical_trials':0,'path_tool_sha256':sha(__file__),
                  'software_manifest_status':'Uncommitted source; tool SHA256 identifies this generator'}
        if graph_mode:
            metadata.update(junction_policy='analytic tangent circular fillets',
                            refinement_note=None, smoothing=None, tangent_yaw='exact analytic derivative')
        (output/'path_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        print(output/'E2_environment.csv')
    except Exception as exc:
        (output/'failure.json').write_text(json.dumps({'error':str(exc),'physical_trials':0,
            'details':getattr(exc,'details',None)},indent=2)+'\n')
        raise
    finally:
        for process in processes:
            if process.poll() is None: process.send_signal(signal.SIGINT)
        for process in processes:
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired: process.kill();process.wait()
        for log in logs: log.close()
        node.destroy_node();rclpy.shutdown()


def main():
    import sys
    if len(sys.argv)>1 and sys.argv[1]=='check-path':
        parser=argparse.ArgumentParser(description='Diagnose a frozen CSV without changing it')
        parser.add_argument('command'); parser.add_argument('--path',type=Path,required=True)
        parser.add_argument('--output',type=Path,required=True)
        parser.add_argument('--speed',type=float,default=.22)
        parser.add_argument('--omega-max',type=float,default=.6)
        parser.add_argument('--map',type=Path)
        args=parser.parse_args()
        print(json.dumps(check_path(args.path,args.output,args.speed,args.omega_max,args.map),indent=2))
        return
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph', action='store_true', help='Round a waypoint polyline without NavFn or a smoother')
    parser.add_argument('--corner-radius', type=float, default=.3)
    parser.add_argument('--map',type=Path,required=True)
    parser.add_argument('--start',type=float,nargs=3,required=True,metavar=('X','Y','YAW'))
    parser.add_argument('--goal',type=float,nargs=3,required=True,metavar=('X','Y','YAW'))
    parser.add_argument('--via',type=float,nargs=2,action='append',metavar=('X','Y'))
    parser.add_argument('--align-start-yaw', action='store_true',
                        help='Record the generated initial tangent as the robot start yaw')
    parser.add_argument('--facing',type=float,nargs=5,metavar=('X1','Y1','X2','Y2','YAW'))
    parser.add_argument('--transition',type=float,default=.35)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path)
    args=parser.parse_args()
    if args.config is None:
        local=Path(__file__).resolve().parents[1]/'params/dwvp_access_path.yaml'
        if not local.exists():
            from ament_index_python.packages import get_package_share_directory
            local=Path(get_package_share_directory('dwpp_test_simulation'))/'params/dwvp_access_path.yaml'
        args.config=local
    generate(args)


if __name__=='__main__':
    main()
