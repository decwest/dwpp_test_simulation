# DWVP Access hardware recording protocol

## One-terminal entry points (2026-10-07)

In the sourced HSR container workspace, `./dwvp_access.sh mapping` starts SLAM
Toolbox, mapping RViz and the existing F310 profile. Release LB, stop, and press
Enter in that terminal to stop the joystick, verify stationary sensors, save the
map, and close the owned launch processes. Maps use `maps/lab_<JST timestamp>/`;
same-second names receive a suffix. Successful saves atomically update
`maps/latest.json`. Save failures leave SLAM running for an Enter-triggered retry;
Ctrl-C cancels without an implicit save. The free threshold is explicitly 0.196.

`./dwvp_access.sh experiment` chooses that last successfully saved map, snapshots
it, and starts AMCL, RViz and F310. Position the robot, set RViz's initial pose,
verify map/scan agreement, release LB and press Enter. The workflow stops teleop,
captures the start, prepares a new bidirectional E1 schedule, checks it, executes
`batch --start-from-current`, and summarizes. The default is all 55 E1 trials;
`--repeats 1` selects 11. `--conditions E1_lateral` restricts conditions. E2 remains
in the manual map-fixed workflow below. The fixed alignment targets and measured
per-trial reference placement are preserved.

By default, each invocation starts fresh in
`results/dwvp_access/<environment>/run_<JST timestamp>/session/`. Logs and the map
snapshot live beside `session/`; `workflow.json` records completion/failure. The
workflow never overwrites old sessions or automatically repeats failed scored trials. Existing
mapping/navigation or nonstandard teleop blocks startup. Standard robot teleop is
handed off as described below.
Keyboard interruption or failure stops the owned batch before localization closes.
`--map PATH` overrides map selection; `--dry-run` starts no nodes and creates no
session. `--no-rviz` / `--no-joy` support isolated headless verification.

The one-terminal workflow enables `batch --continue-on-endpoint-failure`:
an action-successful, freshly verified physical stop outside the original endpoint
tolerances remains a **failed trial**, then the schedule continues. It does not
change tolerances, controller parameters or recorded success. Aborts, timeouts,
stale sensors, unverified stops and alignment failures still stop the workflow.

Use `./dwvp_access.sh experiment --resume [SESSION] --dry-run` to inspect pending
trials, then omit `--dry-run` to localize on the saved map and press Enter to
continue. With no session argument, `results/dwvp_access/latest.json` selects the
previous experiment. Resume validates frozen input hashes and the batch's map
hashes. It preserves recorded successes and verified stopped endpoint failures,
never retries them, and rejects incomplete/aborted attempts or gaps. New startup
logs go under the original run's `resumes/resume_<JST timestamp>/`. Input overrides
such as `--map`, `--params` and `--conditions` are rejected during resume.

To explicitly repeat failed trials, use
`./dwvp_access.sh experiment --retry-failed [SESSION] --dry-run`, then omit
`--dry-run` and set localization before pressing Enter. The source schedule must
be fully recorded; only verified stopped endpoint failures are eligible. Source
records and aggregates are preserved. A separate attempt goes under
`<original run>/retries/retry_<JST timestamp>/session/`, with copied frozen inputs,
source manifest/result snapshots and hashes. Successful source trials are omitted.
Each retry starts from its freshly measured stopped pose, without returning to
the old start or rotating to the old heading, including when resuming a retry
created by an older version. Controller settings and local path geometry are
preserved. Source direction labels identify the original attempts; they do not
command the old map heading. Captures/results record
`start_policy: current_pose_with_turnaround` and `source_direction`. The first leg
uses the operator's stopped pose. Before each subsequent leg, turn in place to
reverse the preceding recorded path's travel direction, then recapture the pose.
Orientation trials change body yaw during translation, so adding pi to the final
body yaw would point the wrong way. On resume, compare the first pending leg's
current XY with the preceding trial's recorded stopped XY. Within 0.5 m, resolve
the return direction from the recorded path, preventing an extra reversal after
interruption. If farther away, use the operator's current position and heading
without a turnaround, checking the full path with the same 3 cm clearance margin.
This exception applies only to the first pending leg on resume; subsequent legs
turn back normally. Missing or invalid previous stopped poses reject this choice.
Record `start_mode: relocated_current_pose` and the distance in the turnaround
diagnostic and start capture. Localization must be aligned before pressing Enter.
For normal turnarounds, search within
+/-5 degrees of that nominal return heading in 1-degree steps, nearest first,
without changing the current XY or the frozen local trial geometry. Candidates
and stopped placements must clear the map with a 3 cm margin beyond robot radius.
Turnarounds use DWVP with a separate 0.005 rad goal checker and a stopped-heading
check of 0.05 rad. If the measured stopped path is blocked or heading exceeds that
bound, recapture and correct in place, with at most three turns total. A failed
action, stale sensor or unverified stop never qualifies for correction. No clear
candidate or exhausted corrections stops before the next scored trial.
Turns are recorded separately under `batches/<timestamp>/turnarounds/`, with unique
attempt IDs; scored trials keep their frozen general goal checker. Candidate
headings, captures, blocking cells and stopped paths are saved in `turnaround_checks/`.
The next trial uses the same stopped capture that passed the final map check.
Rejected placements retain their capture, path CSV and blocking cell coordinates
under `batches/<timestamp>/path_checks/`, without reserving a scored trial. Choose
an open current start position/heading and resume. No corrective motion is sent
to imitate the historical start.
Retry aggregates describe only that attempt, without replacing original failures
or the original success-rate denominator. No input overrides or simultaneous
`--resume` are allowed. Without SESSION the latest experiment pointer is used;
starting a retry updates that pointer, so another invocation does not repeat its
successful trials. Zero failures means no nodes or goals. An interrupted retry
can use `--resume` under the same completed-prefix checks as other sessions.

When standard JOY publishers are present, `mapping` and `experiment` first verify
0.5 s of stationary wheel odometry with fresh arrivals and progressing source stamps,
then use SSH to stop the robot's standard
`joy_linux_node` and `joystick_control_node`. Only exact executables whose parent
is `boot_app.launch.py` and whose cgroup matches the selected robot container are
eligible. PID identity is rechecked and SIGINT is sent through pidfds. Parent
launches and hardware drivers remain running. Startup proceeds only after the
processes exit and the ROS graph has no existing `/joy` or velocity publishers.
Other conflicts, SSH failure or a failed stop prevent starting PC JOY. The robot's
startup files are unchanged and its JOY is not automatically restarted on exit.

Defaults match the workspace's robot login: `HSR_IP` (fallback 192.168.50.10),
administrator, docker.humble.robot.service. Use `--robot-host`, `--robot-user`,
`--robot-container`, or `HSR_SSH_PASSWORD` to override them. Passwords are passed
through sshpass's environment, never command arguments; an empty password selects
SSH key authentication only. `--no-stop-robot-joy` disables the automatic stop while
retaining all conflict checks. No SSH is performed without JOY conflicts or during
dry-run. Handoff reports go to `log/robot_teleop/` and the session's `workflow.json`.
Clock skew is reported but does not block stopping existing input processes; map
capture and experiment motion retain their existing source-clock freshness limits.

The standalone `ros2 launch ytlab2_hsr_modules mapping_joy.launch.py` also combines
SLAM, RViz and F310, but timestamped saving belongs to the shell entry point.
Rebuild `hsrb_mapping`, `ytlab2_hsr_modules`, and `dwpp_test_simulation` after adding
these entry points (`./dwvp_access.sh build` includes them).

## Measured starts and automatic bidirectional trials

Start localization before freezing a session. Stop other navigation and teleop
processes, initialize AMCL in RViz, and park at the desired near-side start.

```bash
./dwvp_access.sh localize map:=/home/dev/ros2_ws/maps/lab01/map_nav2.yaml
# In another terminal, while stationary:
./dwvp_access.sh prepare --output results/dwvp_access/auto01 \
  --params src/third_party/dwpp_test_simulation/params/hsrb_dwvp_access_params.yaml \
  --start-from-current --bidirectional --conditions E1
./dwvp_access.sh batch --session results/dwvp_access/auto01 \
  --map maps/lab01/map_nav2.yaml --dry-run
# Sends motion goals and records all 55 E1 legs:
./dwvp_access.sh batch --session results/dwvp_access/auto01 \
  --map maps/lab01/map_nav2.yaml
```

The captured pose is the forward robot start for every E1 condition. The lateral
reference starts 0.5 m to its right. Reverse legs rotate the canonical E1 geometry
by pi, preserving the local initial error and orientation ramp; their starts are
frozen at the far end. E2 keeps its map coordinates, reversing position order and
adding pi to tangent headings for reverse trials. Supply an E2 CSV and omit
`prepare --conditions E1` for all 80 trials. Conditions/repeats can be restricted
at preparation; selection must preserve the frozen forward/reverse alternation.

Both directions are scored and summarized separately. Only turning and positioning
between legs are excluded, stored under `batches/<run>/alignments/`. The batch
keeps AMCL running, restarts its controller/smoother/planner when parameters change,
checks live map identity and command ownership, and monitors sensor freshness.
Failure or interruption cancels the active goal, allows the smoother to reach zero,
and stops without retrying or advancing. It stops at the last scored endpoint;
an odd number of legs does not add an extra return. Existing completed trial IDs
cannot be reused. The older non-bidirectional mode retains unscored returns.

After an action succeeds, the recorder waits for smoothed zero output and 0.5 s
of stationary, fresh wheel odometry, then checks the latest TF against the original
0.1 m / 0.3 rad tolerances. The wait is bounded using the smoother's deceleration
limits. `settling.csv` and `result.json` retain this check separately from the
action duration and tracking metrics. A user-run HSR outbound trial succeeded,
but its subsequent turn was previously rejected at 0.320 rad while still slowing;
the recorder had evaluated its cached TF immediately after the success response.

Use `batch --resume --dry-run` to inspect the remaining frozen schedule and
`batch --resume` to execute it. Resume preserves only a contiguous successful
prefix whose input hashes match, aligns from the current pose to the next start,
and records that positioning in a new batch directory. Failed/incomplete scored
trials and gaps are rejected; no existing trial or failed alignment is overwritten.
Adding `--continue-on-endpoint-failure` also accepts a contiguous prefix containing
verified stopped endpoint misses; those results remain failed. Old records without
the machine-readable failure reason must include a matching final fresh, zero-command,
stationary `settling.csv` sample. All other failed/incomplete attempts still block resume.
An already recorded schedule sends no goals. A reverse leg can be first on resume.

### Align, then place each E1 path at the measured stop

For bidirectional E1 sessions, add `batch --start-from-current`. Each leg first
aligns toward its **original frozen start position and heading**, which limits
accumulated displacement. It then captures the actual stopped pose and rigidly
places that trial's reference relative to it. The 2.5 m length, 0.5 m lateral
offset, orientation ramp, methods and acceleration settings remain unchanged.
`prepare --start-from-current` captures once; the batch flag captures every leg.
E2 map-fixed routes do not support this option.

Only these unscored alignments use `endpoint_policy=reanchor_after_stop`: a
successful action, verified stop and fresh pose suffice even when the final
residual exceeds 0.1 m / 0.3 rad. The residual and
`final_pose_within_tolerances` remain in the alignment result. Aborted actions,
stale data and failure to stop still fail. Scored trials retain the original
endpoint limits. If already within the frozen start tolerance, alignment is skipped.

The transfer path and the reference at both the intended and measured starts
are checked against the static map with the robot radius. The actual path is
published to RViz before recording. Dry-run loads the schedule and map; future
measured placements can only be checked immediately before each leg.

Each scored run stores `start_capture.json` (fixed alignment target and measured
pose), `reference.csv` (sent geometry), and hashes in `trial.json`. Metrics use
this measured start and reference. Existing records remain `session_fixed` and
new ones use `per_trial_current_pose`; group output flags
`mixed_reference_policies` when both contribute to the same group. Resume checks
the frozen inputs and recorded geometry hashes without rewriting old results.

For the existing `lab01_auto01` session, five recorded trials succeeded and 50
remain. Keep localization running and stop other navigation/teleop nodes:

```bash
./dwvp_access.sh batch --session results/dwvp_access/lab01_auto01 \
  --map maps/lab01/map_nav2.yaml --resume --start-from-current --dry-run
# Sends motion goals:
./dwvp_access.sh batch --session results/dwvp_access/lab01_auto01 \
  --map maps/lab01/map_nav2.yaml --resume --start-from-current
```

`hsrb_moveit` requires `ament_package()` to generate its environment setup scripts;
the workspace wrapper's build now includes this metadata package.

Freshness checks allow TF and sensor source stamps up to 50 ms ahead of the
receiving PC (`CLOCK_FUTURE_TOLERANCE_S`). Independently synchronized HSR and PC
clocks were initially measured about 13.7 ms apart. On October 7 source stamps
led the PC by up to 24.4 ms, repeatedly resetting stationary capture with the old
20 ms allowance. The past-age limits stay unchanged (0.2 s for TF/odom,
0.5 s for scan preflight, the configured metric age for offline analysis).
The same rule applies to capture, preflight, trial completion, batch supervision,
stationary noise and metrics. Local receive ages and command ordering retain a
zero lower bound. Signed source ages are never clamped; capture, result, noise
and summary JSON record `maximum_future_source_skew_s`. Larger future offsets
still fail and require checking clock synchronization on both machines.
Offline metrics use each result's recorded allowance, falling back to the legacy
20 ms only for older records without that field. The per-trial metric rows retain
the applied allowance and the overall summary reports their maximum; increasing
the live allowance does not reinterpret previous trials.

The following protocol describes the full set of local experimental conditions;
the measured-start and bidirectional options change their map placement, not their
local geometry, limits, or method assignments.

This revision prepares **80 trials**. The user has recorded five successful HSR
trials; a complete physical schedule has not been verified. Each assigned
condition/controller combination has five repeats: lateral alignment (DWPP, DWVP;
10 trials), orientation ramp at nominal, half and quarter acceleration (VP_CLIP, VP_SCALED,
DWVP; 45 trials), and environment routes (RPP, DWPP, MPPI Omni, DWB, DWVP; 25 trials).
Experiment 1 checks simulated properties in a noisy, delayed system; Experiment 2
compares existing local planners. Synthetic checks are integration evidence only.

## Configuration and execution

The executable sources are in `ytlab2_hsr/ros2_ws/src/third_party/dwpp_test_simulation`.
The Japanese operator guide provides the full commands. The flow is
`build → generate E2 path → prepare → launch → preview → preflight → run → summarize`.
In that manual flow, only `run` sends a FollowPath goal and repositioning is manual.
The `batch` flow above also sends goals and automates positioning.

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
`E1_orientation_half` uses (0.11, 0.11, 0.3) and
`E1_orientation_quarter` uses (0.055, 0.055, 0.15) for both signs. The frozen condition
`acceleration_scale` feeds controller rendering, smoother rendering and constraint
metrics through the same `condition_common` helper. Prepare a new session
to include the quarter profile; existing frozen sessions retain their original IDs.
The existing 120 s trial timeout covers the expected approximately 30 s quarter-profile run. All
methods use the same OPEN_LOOP velocity smoother. Humble MPPI lacks newer
per-axis acceleration parameters; its raw output is not assumed to obey those
constraints. MPPI retains ECPP experiment 2's `use_path_orientations=false`;
it is assigned only to the tangent-orientation environment route. RPP/DWPP orientation
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
limited-acceleration trajectory generator. The [MPPI and DWB configuration](dwb_configuration.md)
compares every parameter with ECPP experiment 2 and records the omnidirectional,
HSR, Humble and synthetic-completion adaptations.

The 2026-10-06 follow-up uses the cell-width rule
`0.05 / (0.22 / 30) = 6.818 s` to test DWB at 7.0 s, then 8.0 s.
Both horizons stalled during the final turn with the default Oscillation reset.
DWB uses 8.0 s and a 1.0 s reset. The second follow-up explicitly authorizes
lowering only the stopped threshold from 0.11 to 0.005 m/s as an HSR adaptation.
The threshold is below the per-axis velocity increment, `0.22 / 30 = 0.007333 m/s`,
so stopping within one control cycle is feasible when RotateToGoal requires
zero translation. The previous 0.11 m/s setting completed the route but had
one failed controller call. That stopped-threshold update kept the existing
tests and timing boundaries.
See the configuration document for the candidates and the half-acceleration
limit (13.636 s). DWB remains assigned only to nominal E2; no horizon scaling
is implemented. Full results are recorded in the manuscript's
`docs/hardware/baseline_planner_verification.md`. Both verification scripts passed twice with exit code 0; both DWB recordings had zero failed controller calls. Physical HSR trials remain zero.

## Fixed paths and starts

| Condition | Reference geometry and yaw | Required initial pose |
|---|---|---|
| E1_lateral | Local (0,0) to (2.5,0), yaw=0 | Local (0,0.5,0) |
| E1_orientation_nominal / E1_orientation_half / E1_orientation_quarter | Straight 2.5 m; yaw=0 up to 1.0 m, linear 0→π/2 over 1.0–1.3 m, then π/2; spacing 0.01 m | Local (0,0,0) |
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
failures and partial records. All 16 planned groups remain visible with recorded,
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

The hardware entrypoint first generates eleven 2.5 m trajectories with the simulator
mounted read-only, using `uv run --offline --locked --python 3.11.11 --no-sync`.
It reuses the existing virtual environment and interpreter; override paths with
`DWVP_SIMULATOR_ROOT` or `DWVP_UV_BINARY` when needed. No dependencies are installed.
The HSR Python 3.10 summary reads their saved poses and exact command histories.
The 119 comparisons cover six errors, violation percentage, travel time, evaluation
duration and matching transients; absolute tolerance is 1e-9 in each metric's unit,
relative tolerance zero. Timing is excluded from cross-language numerical equality.
Analytical tests cover irregular timestamps, constant/triangular errors, windows,
gaps, ramp lag/overshoot, transformed paths, unknown bounds and noise statistics.

The hardware entrypoint builds four packages, runs Python 3.10 unit tests,
recorder failure/freshness/cancellation checks, 16 actual-controller goals,
16 actual-controller/recorder integrations, NavFn generation/import and launch
argument inspection. Every assigned combination is exercised once, including all three
orientation acceleration profiles. Unassigned combinations and the wrong lateral
start are rejected. All seven methods must produce per-call timing records.
For each of VP_CLIP, VP_SCALED and DWVP, consecutive raw-controller command
changes and smoother-output changes must be at most (0.11,0.11,0.3)/30 per cycle
for the half profile and (0.055,0.055,0.15)/30 for the quarter profile. The assertion
tolerance is 1e-8 in each velocity component. The existing synthetic action timeout
of 120 s and recorder-process timeout of 160 s cover the quarter profile.
Only Nav2's extra terminal zero suffix, emitted outside the controller, is omitted
from the raw-controller increment assertion; recorded metric streams retain it.
This consecutive-command assertion is distinct from the recorder metric, which
pairs raw commands with the strictly earlier received smoother output.

E2 controller inputs use a synthetic tangent path and free map; the separate
NavFn test checks an obstacle-map detour, fixed endpoints and forward tangents.
A successful ROS goal can still contain scheduling/clock gaps; tests require the
quality counters to remain visible and finite observations to remain in group means.

The current two-round results and source hashes are documented in the manuscript
copy at `docs/hardware/quarter_acceleration_verification.md`. They concern the
**uncommitted working tree with synthetic inputs**, with zero physical HSR trials.
The older synthetic JSON and `method_update_verification.md` remain historical.
Real AMCL, sensors, robot dynamics and the RViz screen have not been validated.
Regenerate `software_manifest.json` after commit; no commit/push occurs in this task.
