# Experiment 2 plan v2 tooling

`params/dwvp_access_experiment.yaml` applies the v2 overrides only to
`E2_environment`. All E1 renders retain the prior settings. PP cost regulation
and rotate-to-heading are disabled in E2; goal approach and lookahead are retained.
MPPI path orientations are enabled and DWB's planar speed cap is 0.22 m/s.
E2 explicitly selects LimitedAccelGenerator, an 8 s horizon, and its prior
spatial rollout defaults. The October 7 robot-session base template retains
StandardTrajectoryGenerator and 1.7 s for unchanged E1 rendering. Generator
timing is resolved after the condition overrides; the E2 dynamic-window period
remains the control period. Static-map walls from the robot-session template
remain active in every condition.
The scored method list may be narrowed for explicit fixed-route reacquisition
without removing controller overrides or the DWVP configuration used for
unscored returns. Existing frozen sessions retain their own configuration.
The selected critic settings and evidence are in the manuscript repository's
`docs/hardware/e2_plan_v2.md` and `e2_plan_v2_verification.json`.

Use the source-checkout wrapper, which always starts Docker with `--network none`:

```bash
export DWVP_E2_WORK=$(mktemp -d /tmp/dwvp-e2.XXXXXX)
export DWVP_E2_INPUT=/absolute/map/directory
scripts/e2_tooling.sh build
scripts/e2_tooling.sh path --map /input/map.yaml \
  --start X Y YAW --goal X Y YAW --output /work/route
scripts/e2_tooling.sh check-path --path /work/route/E2_environment.csv \
  --output /work/check
scripts/e2_tooling.sh sweep --path /work/route/E2_environment.csv \
  --output /work/sweep
# Put the report.json selected overrides into the E2 template, then:
scripts/e2_tooling.sh test
scripts/e2_tooling.sh verify --path /work/route/E2_environment.csv \
  --output /work/verify --repeats 2
```

Outputs must be new directories. `path` invokes NavFn, removes only logged sub-cell
reversals, invokes Nav2's SimpleSmoother action, resamples at 0.02 m, and assigns
forward tangent yaw over s ± 0.10 m (clipped at endpoints). All native smoother
weights and tolerances are retained. Humble 1.1.19 uses four hardcoded refinement
passes; the newer `refinement_num=2` parameter is unavailable. The saved runtime
snapshot and metadata expose this difference. Both Nav2 and the dense circular
footprint check reject colliding paths.

`check-path` reports signed curvature d(yaw)/ds, the yaw-rate demand at 0.22 m/s,
every point exceeding ±0.6 rad/s, spacing and neighboring yaw steps. The PNG also
shows XY segment curvature. It does not automatically change speeds or reject a
CSV for curvature; examine the feasibility report before using a new route.

`sweep` tests MPPI angle weights 6/12/20 (alignment weight 30) and DWB paired
forward distances 0.1/0.325/0.5 crossed with PathAlign scales 32/48. Its
`plan_before_results.json` fixes the selection rule before execution: among
completed settings per controller, minimize the sum of the position and heading
integrals, each normalized by its smallest completed-grid value. Exact ties use
grid order. Zero minima use their limiting score; no completed candidate means
no selection. The common interval ends 0.6 m before the path end.

This is the existing fake robot with real Humble plugins, common smoother,
production recorder and metrics, free scans, and a free map sized to cover the
input path. It is not a physical robot or obstacle-sensing replay. Every trial's
runtime parameter match, completion, errors, constraints, timing and speed/rotation
diagnostics are retained. The maximum trial duration remains 120 s.

Unit comparisons use the simulator revision
`4535f3e2c3d52a36e2916b5768099173251fd2d3`, exported from its read-only Git objects
to the scratch directory. That revision implements the existing evaluation
window. Later simulator changes to full-run metrics are not silently imported
into E1 or E2. No neighboring checkout is changed.

## Shelf-facing route design (2026-10-14)

`path` accepts repeated `--via X Y` arguments. Native NavFn segments are joined
before sub-cell cusp removal and a single SimpleSmoother action, so the via
coordinates guide planning but are not pinned in the final path. Via-junction
reversal cleanup also removes any coincident points it creates. No-via behavior
is covered by `path-test`.

`--facing X1 Y1 X2 Y2 YAW --transition 0.35` selects points whose projection lies
on the finite line segment and whose perpendicular distance is below 0.6 m.
The selected points receive fixed yaw. Wrapped shortest-angle smoothstep ramps
occupy 0.35 m of arc length before and after each selected interval; other points
keep tangent yaw. Empty selection, overlapping ramps and ramps outside the path
are errors. The settings and sampled arc intervals are saved in metadata.

The following commands reproduce P2 using the new map as the read-only input:

```bash
scripts/e2_tooling.sh path --map /input/map.yaml \
  --start 2.8 0.7 3.141592653589793 \
  --via -0.1 0.7 --via -0.1 -2.6 --goal 2.2 -2.6 0 \
  --facing -0.1 -0.8 -0.1 -1.9 3.141592653589793 --transition 0.35 \
  --output /work/P2
scripts/e2_tooling.sh check-path --path /work/P2/E2_environment.csv \
  --output /work/P2_check
scripts/e2_tooling.sh sweep --path /work/P2/E2_environment.csv \
  --config /ws/src/dwpp_test_simulation/params/dwvp_access_e2_route_design.yaml \
  --output /work/P2_sweep
# Merge report.json selected overrides into a new copy under /work, then:
scripts/e2_tooling.sh verify --path /work/P2/E2_environment.csv \
  --config /work/P2_selected.yaml --output /work/P2_verify \
  --repeats 3 --dwvp-fixed-lookahead 0.11
```

`check-path` finds `map.yaml` beside the path or accepts `--map`. Clearance is
centre-to-occupied-cell-rectangle distance, with unknown cells reported
separately; the body margin subtracts the 0.22 m radius. Tables in this map are
represented by their legs, so this does not check tabletop overhang. The report
retains yaw steps and rate exceedance locations, adds geometric minimum radius
and separate tangent-yaw rate diagnostics, and saves a four-panel PNG.

The route-design configuration is separate from the default configuration. It
keeps all E1 settings, scores only VP_CLIP/DWVP/MPPI/DWB in E2, disables PP cost
regulation and rotate-to-heading, and sets DWVP lateral_time_constant=0.6 s.
The extra fixed-lookahead run uses the same DWVP controller ID in a separate
frozen session; `variant=DWVP_fixed` distinguishes it in the combined report.
Verification returns a nonzero exit status when any trial fails, after retaining
all attempted repetitions and their results.

`e2_tooling.sh test` generates both historical metric references and the existing
E1 response counterpart before running all tests. Both Docker invocations keep
`--network none`; simulator code and dependencies remain read-only.
