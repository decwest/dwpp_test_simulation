# DWVP Access hardware recording protocol

This revision prepares **65 trials; physical HSR trials remain zero**. Each assigned
condition/controller combination has five repeats: lateral alignment (DWPP, DWVP;
10 trials), orientation ramp at nominal and half acceleration (VP_CLIP, VP_SCALED,
DWVP; 30 trials), and environment routes (RPP, DWPP, MPPI Omni, DWB, DWVP; 25 trials).
Experiment 1 checks simulated properties in a noisy, delayed system; Experiment 2
compares existing local planners. Synthetic checks are integration evidence only.

## Configuration and execution

The executable sources are in `ytlab2_hsr/ros2_ws/src/third_party/dwpp_test_simulation`.
The Japanese operator guide provides the full commands. The flow is
`build → generate E2 path → prepare → launch → preview → preflight → run → summarize`.
Only `run` sends a FollowPath goal. Repositioning between trials is manual.

`params/dwvp_access_experiment.yaml` is the single source for nominal tuning,
geometry, starts, tolerances and metric thresholds. `prepare` merges those
values into `params/hsrb_dwvp_access_params.yaml`, a controller-specific template,
and writes executable `SESSION/nav2_params_<condition>.yaml` files.
Use `dwvp_access.sh launch SESSION TRIAL` to select the frozen file for that trial;
`parameters --session SESSION --trial TRIAL` prints it. Restart the controller
and smoother when changing acceleration profile. The runtime snapshot checks
reject mismatched controller/smoother limits before sending a goal. Never launch the template
as if it contained the expanded common settings.
Snapshots query ListParameters/GetParameters directly while servicing sensor
callbacks, with a 15 s monotonic deadline per node; no CLI daemon cache is used.

The nominal lookahead is 0.75 s, bounded by 0.11 and 0.33 m, with 30 Hz
control. The component velocity bounds are ±(0.22, 0.22, 0.6), and component
acceleration/deceleration magnitudes are (0.22, 0.22, 0.6), in SI units.
`E1_orientation_half` uses (0.11, 0.11, 0.3) for both signs. The frozen condition
`acceleration_scale` feeds controller rendering, smoother rendering and constraint
metrics through the same `condition_common` helper. All
methods use the same OPEN_LOOP velocity smoother. Humble MPPI lacks newer
per-axis acceleration parameters; its raw output is not assumed to obey those
constraints. MPPI PathAlignCritic uses path orientations. RPP/DWPP orientation
errors along the path are marked `reference_only`.

VP_CLIP selects the existing component-clipping branch of the DWVP plugin by
setting `use_dynamic_window_vector_pursuit=false`. That branch clips against
the regulated dynamic window before the external smoother. It is not an
unconstrained vector sent directly to the smoother. `vp_translation_speed<=0`
selects the minimum of positive and reverse planar speed limits. Positive values
set the desired magnitude and the orientation travel-time denominator. DWVP
always chooses the largest intersecting alpha, or the closest point to the ray
with larger alpha breaking distance ties, and allows alpha > 1. DWVP, VP_CLIP and VP_SCALED keep `approach_velocity_scaling_dist=0.6` m. Goal approach regulation and
terminal control provide slowdown; disabling approach regulation can produce
overshoot (0.092 m in the nominal Python straight-path check, not a physical HSR
measurement). The obsolete plugin CSV column `is_accel` has been removed.
Output is scale-invariant only while the entire desired vector
scales uniformly: activating `min_orientation_time` can change the ray direction.
The simulator accepts explicit zero as zero translation and rejects negative
values; the hardware interface follows the requested automatic-selection rule.

VP_SCALED sets `use_dynamic_window_vector_pursuit=false` and
`use_uniform_velocity_scaling=true`. It scales all components together to the
regulated velocity box (without acceleration), then clips to the common regulated
dynamic window. It shares VP_CLIP's terminal handling. DWB uses an omnidirectional
limited-acceleration trajectory generator; [DWB configuration](dwb_configuration.md)
lists all departures from the Humble bringup baseline and the reasons.

## Fixed paths and starts

| Condition | Reference geometry and yaw | Required initial pose |
|---|---|---|
| E1_lateral | Local (0,0) to (2.5,0), yaw=0 | Local (0,0.5,0) |
| E1_orientation_nominal / E1_orientation_half | Straight 2.5 m; yaw=0 up to 1.0 m, linear 0→π/2 over 1.0–1.3 m, then π/2; spacing 0.01 m | Local (0,0,0) |
| E2_environment | Saved NavFn positions, smoothed, with forward tangent yaw | Map pose in settings, otherwise planner metadata start, otherwise first CSV pose |

E1 poses are transformed by the fixed `--origin`. E2 CSVs and obstacle survey
shapes already use map coordinates and receive no second origin transform.
`preflight` and `run` check the condition start, not the first path point, with
0.1 m and 0.3 rad tolerances. Thus a robot at the E1_lateral path origin is rejected.
The preflight also checks current TF, odometry receive/source age, laser source
age and laser TF, and all supplied runtime parameters of the selected controller,
checkers and smoother. Start pose is rechecked after sensor discovery.

Each repeat contains one block per condition; the assigned methods are shuffled
within that block using the saved seed. All methods receive the same path CSV.
The session stores expanded and source settings, paths and hashes, map origin,
condition starts, seed and trial order. `conditions.<ID>.methods` defines valid
assignments; unknown/unassigned combinations and profile substitutions are rejected
before ROS initialization. No override flag is provided. Schema 2 sessions must
be regenerated as schema 3; docking is no longer a measured condition. Sessions and attempted trial directories
are created exclusively: no retry overwrites a prior attempt. An incomplete
attempt remains counted. A new setting requires a new session.

## Offline E2 generation

`scripts/dwvp_access_path.py` requires Docker `--network none` and
`ROS_LOCALHOST_ONLY=1`. It starts map server, NavFn planner server, lifecycle
manager and a static map-to-base TF, and calls ComputePathToPose with
`use_start=true`. It does not start a controller or connect to the robot.
Action availability alone does not mean the static layer is ready. Before sending
the goal, the generator waits for a published global costmap matching the input
map dimensions/resolution and containing known cells (30-second timeout).
The generator starts node executables directly and waits for them to terminate
on exit, so planner, map and static-TF processes do not remain for later runs.

The default smoother first resamples at 0.02 m arc-length intervals, then applies
elastic position smoothing with data weight 0.2, smoothing weight 0.3, at most
200 iterations and tolerance 1e-6 m. Each simultaneous interior update is
`0.2*(original-current) + 0.3*(previous+next-2*current)`; endpoints stay fixed.
`params/dwvp_access_path.yaml` selects `elastic` or `none` and the strengths.
Central secants determine forward tangent yaw; endpoint yaw uses a one-sided
secant. Requested goal yaw is retained in metadata, but final CSV yaw is tangent.
This calculation does not use Nav2 smoother orientation inference.

The final path is checked, including between samples, with a circular footprint
against occupied, unknown and outside-map cells. Invalid smoothing is rejected,
not silently reduced in strength. Supported maps are trinary grayscale images.
This is a map-based geometric check with a configured radius, not surveyed
clearance or physical validation.

Outputs are `E2_environment.csv` (x,y,yaw in m/rad), `preview.png`, original
NavFn positions, planner/smoothing parameters, map copies and metadata containing
CSV, parameter and map hashes. Failure logs remain in the reserved output folder.
Inspect the preview before importing the CSV. `prepare` checks matching metadata
when it is alongside the CSV and copies it into the session.

## Recorded streams

| File | Contents |
|---|---|
| reference.csv, trial.json | Exact map-frame reference; trial and configuration provenance |
| *_runtime.yaml | Actual controller server and smoother parameter snapshots |
| raw.csv | Subscribed controller output on /cmd_vel_nav |
| applied.csv | Subscribed smoother output on /omni_base_controller/cmd_vel |
| odom.csv | Wheel odometry velocities, source and receive timestamps |
| tracking.csv | Map pose at control frequency, TF/stream/source ages, laser minimum valid range, wheel speed |
| timing.csv | Controller ID, sequence, ROS call-start stamp, receive stamp, steady-clock duration, success/exception |
| result.json | Status, duration, final pose errors, cancellation and sensor checks |

`applied` means the command published by the smoother, not confirmation that
the motor driver applied it at that instant. Twist streams have ROS and monotonic receive stamps;
odometry and laser also have source stamps. Timing duration is measured at the
call site, not estimated from message arrival intervals.

## Metrics and quality accounting

`summary.json` and `trial_metrics.csv` retain every attempted trial, including
failures and partial records. All 13 planned groups remain visible with recorded,
pending, success, quality-qualified and evaluation-complete counts. Each metric
uses all finite observed values and reports mean, sample SD (n−1), and its own n.
Flags never discard a trial; missing values remain null. SD is null for n < 2.
Travel time alone requires successful completion. Quality-qualified success counts
retain the previous freshness/startup criteria as diagnostics, not as exclusions.

The common columns are maximum position error [m], time-mean position error [m],
position-error integral [m·s], maximum heading error [degrees], time-mean heading
error [degrees], heading-error integral [degree·s], command violation [%], travel
time [s], and mean controller-call time [ms]. Position error is distance to the
closest segment; heading error uses wrapped yaw interpolation at that same
projection. RPP/DWPP heading errors retain the `reference_only` label in JSON.

Each condition freezes `evaluation: {start_m: 0.0, goal_margin_m: 0.6}` in its
configuration. E1 evaluates 0–1.9 m of its 2.5 m straight path; E2 evaluates
projected arc length from zero to total path length minus 0.6 m. Progress extends
past path endpoints to exclude samples behind the start. A path shorter than the
window is rejected during preparation. Fresh, finite map poses require TF and
odometry receive/source ages in [0,0.2] s. An integral uses trapezoids between
adjacent in-window, fresh samples and their actual recorded time difference.
Excluded samples are never bridged and window boundaries are not interpolated.
Nonpositive/nonfinite time intervals are counted and omitted. The mean equals
integral/evaluated duration. A singleton has zero integral and undefined mean;
an empty window has no error measurements. Endpoint yaw alignment is excluded
from these errors. Duration remains acceptance-to-terminal ROS time, including
final rotation; success still requires action success and a fresh final pose
within 0.1 m and 0.3 rad, without an additional stop test.

Only E1_lateral adds `crossing_m`, the largest signed perpendicular error on the
side opposite the configured initial offset, with no deadband subtraction.
Only the orientation conditions add `transition_heading_lag_deg` (maximum
reference-minus-actual during the yaw ramp) and
`post_transition_heading_overshoot_deg` (maximum actual-minus-final-reference
strictly after the ramp). Both are nonnegative and use the same evaluation
window. Ramp boundary comparisons allow 1e-10 m for map-transform rounding.
Legacy full-trial RMSE, lead/lag, deadband crossing and convergence diagnostics
remain in per-trial files; they are absent from aggregate tables.

`summarize` writes `tables/<condition>.csv`, `.md`, and `.note.txt`. Every table
uses the common columns with mean, sample SD and n; transient columns appear only
for the matching condition. If every method has recordings, all recorded rates
are defined, and all are zero, the rate column is omitted and a sentence states
that classified cycles have 0% violations. Unknown fractions and bounds remain
visible even in that case. No-data groups never imply 0%.

For 1b, `runs/<trial>/orientation_timeseries.csv` contains time, projected
progress, signed yaw error, window/ramp membership, freshness, wheel speed,
running minimum speed within the ramp, and the constant prediction
`omega_max / (abs(goal_yaw)/ramp_length) = 0.114591559 m/s`.
The last running-minimum entry is the ramp minimum. Missing/stale speeds are blank.
Speed minima and predictions are not columns in the aggregate tables.

## Command constraints and unknown cycles

The existing velocity/increment test and normalized tolerance 1e-6 are retained.
For each raw command u and axis i, velocity excess is
`max(u_i-upper_i, lower_i-u_i, 0)/max(abs(lower_i),abs(upper_i))`.
For d=u−v, positive increments use `max_accel`, negative increments the magnitude
of `max_decel`, with excess `max(abs(d_i)/(a_i/f)-1,0)`. Condition acceleration
scale feeds these bounds, controller and smoother from the same frozen settings.
The reference v is the strictly earlier received smoother output within 0.2 s;
no zero or previous raw command is substituted. Applied denotes smoother output,
not motor-side confirmation. A velocity violation alone can classify a cycle
when its increment reference is unavailable. Invalid/latest stale applied values
and invalid raw commands remain explicitly counted.

For V violating, K classified, U unknown, and N=K+U received cycles:

- `constraint_violation_pct = 100 V/K`;
- `constraint_unknown_samples = U`, `constraint_unknown_pct = 100 U/N`;
- `constraint_violation_lower_pct = 100 V/N`;
- `constraint_violation_upper_pct = 100 (V+U)/N`.

`constraint_unknown_flag` marks U/N above
`metrics.maximum_unknown_command_pct` (default 5%). Equality is not flagged.
A zero denominator is mathematically undefined: the value remains null and
`constraint_no_evaluable_cycles` identifies it; with N>0, K=0 the bounds are
0–100%. Finite rates remain in averages despite unknown cycles or other flags.

New event CSVs keep both ROS `stamp_s` and `receive_monotonic_s`. Monotonic receive
time determines ordering/freshness for command pairing, avoiding wall-clock jumps.
ROS time still selects the action interval. Historical files without monotonic
columns use ROS receive time. Diagnostics distinguish no preceding output, stale
output, invalid values, receive gaps, and ROS-versus-monotonic clock steps (>0.1 s).
The first raw command can precede the smoother's first output because the smoother
stops publishing after its idle timeout. It remains unknown. Transport ordering
and independent timers still limit this received-command metric.
Per-axis excess maxima, nominal exceedance durations (count/f), and smoother
input/output differences remain secondary diagnostics. No lost DDS message is
reconstructed and none of these measurements certify physical acceleration.

## Stationary localization noise

With the robot manually stopped, run:

```bash
./dwvp_access.sh stationary-noise --output /tmp/dwvp_noise_session01 --duration 30
```

This command only subscribes to TF and wheel odometry; it publishes neither a
velocity nor a goal. `--frame`, `--base-frame`, `--odom-topic`, `--frequency`, and
`--max-age` select observation settings. `noise_samples.csv` preserves sampled
poses, source/receive ages, speed and acceptance. Duplicate TF stamps are excluded.
`noise_summary.json` and `.csv` report sample SD of x and y, combined position SD
`sqrt(var(x)+var(y))`, maximum radial distance from the mean position, axis ranges,
and SD/maximum absolute deviation/range of unwrapped heading in degrees.
The defaults reject stale samples older than 0.2 s. Any fresh observation above
0.005 m/s or 0.01 rad/s, or fewer than two usable poses, makes the command exit 1
with `stationary_verified=false`; raw records remain available. Rejected and
moving counts are explicit. Localization samples can be temporally correlated.
Report this measured noise beside 1a crossing, without subtracting it from the
crossing metric. This task records only synthetic noise checks, not HSR noise.

Legacy E2 near-laser speed and surveyed clearance remain per-trial diagnostics;
they are not additional common table columns.

## Controller computation time

`dwpp_test_simulation::TimedController` delegates all lifecycle, plan and speed
limit calls to `wrapped_plugin` under the original controller name. Each of the
seven controllers is wrapped identically. The wrapper takes steady-clock readings
immediately around `computeVelocityCommands()` and publishes a DiagnosticArray
on `/dwvp_access/controller_timing` after stopping the clock. It records exceptions
and rethrows them; duration does not include message construction or publication.
Sequence numbers expose internal gaps. Recorder summaries include missing data,
failed calls, mean, p95, maximum and the difference between raw message count and
successful timing count. Boundary/terminal zero messages can produce a small
count difference; leading/trailing loss cannot be uniquely inferred from that
count. Available finite timing observations remain in aggregate means; gaps, invalid
samples and raw-count mismatches remain diagnostics alongside those means.

This follows the measurement boundary used by the inspected ECPP controller-server
instrumentation. The wrapper avoids modifying upstream Nav2 packages and follows
the [Humble controller interface](https://github.com/ros-navigation/navigation2/blob/humble/nav2_core/include/nav2_core/controller.hpp).
It includes work performed inside the plugin, including its transforms/collision
checks and waits. It excludes server-side pose/odom acquisition, progress/goal
checks outside the call, costmap waiting, DDS publication and sleep. OS scheduling
can increase the measured wall duration. Diagnostic publication adds unmeasured
cycle overhead. It is therefore **per-call wall time, not full control-loop time,
CPU time or a real-time guarantee**. Native steady-clock C++ timings and Python
simulation timings are not directly comparable.

## Comparison with the current Python simulator

This table was checked against `metrics.py`, `simulation.py` and
`access_metrics.py` after the four-study reorganization.

| Quantity | Python simulation | Hardware recorder |
|---|---|---|
| Geometric reference | Closest segment with wrapped yaw interpolation | Same geometry; duplicate positions rejected |
| Tracking statistic | Maxima, adjacent-sample absolute-error trapezoids and time means in the spatial window | Same definitions, excluding stale/missing samples; 0–1.9 m for E1 |
| Goal tolerances | Default 0.02 m and 1 degree | 0.1 m and 0.3 rad |
| Success/time | Goal tolerance plus component stop threshold 0.001; study rejects collision; simulated steps | Action success plus fresh terminal pose; acceptance-to-result elapsed time |
| Constraint increments | Exact previous simulated applied command and fixed dt | Strictly earlier received smoother command, with age gate and missing counts |
| Exceedance percentage | Exact simulated command cycles | Violations per classified received cycle, unknown counts/fraction and all-cycle bounds |
| Exceedance duration | Per-step duration, absolute tolerance 1e-10 | Per-received-cycle duration, normalized tolerance 1e-6 |
| Lateral crossing | Raw opposite-side crossing in evaluation window | Same; legacy deadband diagnostic stays outside tables |
| Lateral convergence | First and sustained 2% band; distance integrates applied speed | First 10% band; sampled map-pose distance, no inference across stale prefix |
| Computation time | Timed Python controller call, microseconds | Delegated C++ Nav2 call, milliseconds, including exception samples |
| Near-obstacle speed | Known obstacle surface distance and simulated applied speed | Laser minimum range and measured wheel-odometry speed |
| Clearance | Simulated swept circular footprint and known shapes | Only supplied surveyed shapes, sampled map poses and circular radius |
| Aggregates | All finite observations with per-metric n | Same, retaining quality flags; successful travel time only |

## Revalidation and revision provenance

Run both entrypoints from their package directories. All builds and validation
run inside Docker `--network none`, with read-only sources and output under `/tmp`:

```bash
# nav2_omnidirectional_dwvp_controller
DWVP_VERIFY_WORKSPACE=$(mktemp -d /tmp/dwvp-plugin-validation.XXXXXX) \
  DWVP_HUMBLE_IMAGE=docker-hsr:latest ./scripts/verify_humble.sh
# dwpp_test_simulation
DWVP_ACCESS_VERIFY_ROOT=$(mktemp -d /tmp/dwvp-access-validation.XXXXXX) \
  DWVP_HUMBLE_IMAGE=docker-hsr:latest ./scripts/verify_hardware_tooling.sh
```

The hardware entrypoint first generates eight 2.5 m trajectories with the simulator
mounted read-only, using `uv run --offline --locked --python 3.11.11 --no-sync`.
It reuses the existing virtual environment and interpreter; override paths with
`DWVP_SIMULATOR_ROOT` or `DWVP_UV_BINARY` when needed. No dependencies are installed.
The HSR Python 3.10 summary reads their saved poses and exact command histories.
The 86 comparisons cover six errors, violation percentage, travel time, evaluation
duration and matching transients; absolute tolerance is 1e-9 in each metric's unit,
relative tolerance zero. Timing is excluded from cross-language numerical equality.
Analytical tests cover irregular timestamps, constant/triangular errors, windows,
gaps, ramp lag/overshoot, transformed paths, unknown bounds and noise statistics.

The hardware entrypoint builds four packages, runs Python 3.10 unit tests,
recorder failure/freshness/cancellation checks, 13 actual-controller goals,
13 actual-controller/recorder integrations, NavFn generation/import and launch
argument inspection. Every assigned combination is exercised once, including both
orientation acceleration profiles. Unassigned combinations and the wrong lateral
start are rejected. All seven methods must produce per-call timing records.
For the three half-acceleration VP methods, consecutive raw-controller command
changes and smoother-output changes must be at most (0.11,0.11,0.3)/30 per cycle.
Only Nav2's extra terminal zero suffix, emitted outside the controller, is omitted
from the raw-controller increment assertion; recorded metric streams retain it.
This consecutive-command assertion is distinct from the recorder metric, which
pairs raw commands with the strictly earlier received smoother output.

E2 controller inputs use a synthetic tangent path and free map; the separate
NavFn test checks an obstacle-map detour, fixed endpoints and forward tangents.
A successful ROS goal can still contain scheduling/clock gaps; tests require the
quality counters to remain visible and finite observations to remain in group means.

The current two-round results and source hashes are documented in the manuscript
copy at `docs/hardware/metrics_alignment_verification.md`. They concern the
**uncommitted working tree with synthetic inputs**, with zero physical HSR trials.
The older synthetic JSON and `method_update_verification.md` remain historical.
Real AMCL, sensors, robot dynamics and the RViz screen have not been validated.
Regenerate `software_manifest.json` after commit; no commit/push occurs in this task.
