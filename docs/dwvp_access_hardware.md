# DWVP Access: HSR recording protocol

Status: **prepared, not recorded**. The synthetic ROS test verifies the recorder,
not HSR tracking performance. This protocol requires ROS 2 Humble, a localized
HSR, and the renamed `nav2_omnidirectional_dwvp_controller` plugin.

Build and source the updated workspace before using the new package name:

```bash
cd /path/to/ytlab2_hsr/ros2_ws
colcon build --packages-up-to dwpp_test_simulation
source install/setup.bash
```

The HSR workspace migration is on `feature/dwvp_access`. The experiment package
also has its own `feature/dwvp_access` branch, which must be checked out inside
the submodule. The parent workspace retains its previous experiment-package
pointer rather than incorporating unrelated earlier local changes.

## Experimental conditions

| Task | Reference | Controllers | Repeats |
|---|---|---|---:|
| A_path1 | IROS 1 m + 1 m right-angle XY, tangent yaw | RPP, DWPP, MPPI, DWVP | 5 |
| A_path2 | IROS 0.75 m-amplitude, 1.5 m-long, 1.5-cycle cosine XY, tangent yaw | RPP, DWPP, MPPI, DWVP | 5 |
| A_obstacle | Surveyed fixed route through a static environment, tangent yaw | RPP, DWPP, MPPI, DWVP | 5 |
| B_path1 | Same XY as A_path1, constant local yaw 0 | MPPI, DWVP | 5 |
| B_path2 | Same XY as A_path2, yaw `(pi/2)*(3*u*u-2*u*u*u)`, `u=s/total_length` | MPPI, DWVP | 5 |

There are **80 runs**. Ordering is shuffled within each repeat with a saved seed.
The orientation for A_path1 intentionally differs from the old IROS final-yaw
change. The original IROS inputs remain archived separately. Both controllers
receive the identical pose sequence in B; no tangent-yaw reconstruction occurs.

Defaults: 30 Hz, body-axis limits ±0.22 m/s and ±0.60 rad/s, axis acceleration
magnitudes 0.22 m/s² and 0.60 rad/s², goal tolerance 0.1 m and 0.3 rad, timeout
120 s. These are component limits; the diagonal translational speed can exceed
0.22 m/s. The nominal DWVP norm ceiling is 0.32 m/s, above the box corner.
The configured circular collision footprint has radius 0.22 m; verify against
the experiment's physical robot configuration before collecting data.

The MPPI Omni profile uses supplied path orientations, with PathAngle and
PreferForward critics disabled. It is shared by A and B. Humble MPPI does not
implement the newer `ax_max/ay_max/az_max` controls. All methods therefore pass
through the same velocity smoother. Record both sides of that smoother.
The YAML contains **initial pilot settings**, not a claim that tuning is complete.
After pilot checks, freeze one profile before all recorded comparisons.

## Freeze a session

The static-obstacle reference must be surveyed on site. Supply a CSV with the
header `x,y,yaw`, in metres/radians, in the same local frame as the IROS routes.
Specify the common local-frame origin in the map. Do not realign every method
to its own starting pose, because that would hide initial-condition differences.

```bash
ros2 run dwpp_test_simulation dwvp_access_experiment.py prepare \
  --output /data/dwvp_access/session01 \
  --params /path/to/frozen/hsrb_dwvp_access_params.yaml \
  --origin MAP_X MAP_Y MAP_YAW \
  --obstacle-path /path/to/surveyed_obstacle_path.csv
```

Omit `--origin` and `--obstacle-path` to generate a planning manifest without
authorizing a run. The runner refuses an unset origin or an undefined route.
Preparation never starts Nav2 or moves the robot. The session contains exact
paths, configuration hashes, and the full trial list in `manifest.json`.
Existing session directories cannot be overwritten.

## Start the experiment stack

Stop the previous Nav2 stack before starting this dedicated stack. Keep the
HSR driver and sensors running. Use the actual map of the surveyed environment.

```bash
ros2 launch dwpp_test_simulation dwvp_access_hsr.launch.py \
  params_file:=/data/dwvp_access/session01/nav2_params.yaml \
  map:=/path/to/surveyed/map.yaml
```

This launch starts localization, controller_server, velocity_smoother and their
lifecycle manager. It does not start a GUI, global planner, or automatic trials.
Its command chain is explicit:

`controller_server -> /cmd_vel_nav -> velocity_smoother -> /omni_base_controller/cmd_vel`

This differs from the older TMC launch, which publishes directly to the robot.
Do not run both stacks concurrently. Check the graph once during the pilot and
ensure the applied topic has only the intended smoother publisher.

## Record a trial

Reset the HSR to the fixed initial position and the first reference yaw for the
selected task. Read the next ID from the shuffled manifest. Each invocation
sends exactly one FollowPath goal. There is no automatic physical repositioning.

```bash
ros2 run dwpp_test_simulation dwvp_access_experiment.py run \
  --session /data/dwvp_access/session01 --trial B_path1_DWVP_r1
```

The runner checks fresh map TF and wheel odometry, start-pose tolerance, frozen
input hashes, and all explicitly configured parameters of the selected controller,
goal/progress checkers, and velocity smoother before sending a goal. Odometry
freshness uses both receipt and source timestamps. Parameter collection continues
servicing subscriptions, and the start pose is checked afterward.
It uses the same `general_goal_checker` for all methods. Goal rejection,
timeout, action failure, or a final pose outside tolerance are failures.
Completion time includes terminal rotation. Ctrl-C and timeout request goal
cancellation and separately record acknowledgement and terminal-state confirmation.
If confirmation is unavailable, the result explicitly reports that condition.
Failed runs stay in the dataset; use a new session for a changed configuration.

Each trial has:

- `reference.csv`, `trial.json`: exact map-frame pose reference and provenance.
- `controller_server_runtime.yaml`, `velocity_smoother_runtime.yaml`: actual settings.
- `raw.csv`: subscribed controller commands, all three body components.
- `applied.csv`: subscribed smoother output, not a calculated clipping estimate.
- `odom.csv`: measured wheel odometry, with receive and source timestamps.
- `tracking.csv`: map pose at 30 Hz, TF/stream ages, and elapsed time.
- `result.json`: action status, final errors, success, duration, or failure reason.

The trial directory cannot be overwritten. Command CSV times are ROS receive
times; middleware timing jitter must not be interpreted as actuator acceleration.
Map pose and wheel odometry describe different measured quantities. Neither
can be replaced by the requested command in analysis.

## Analyze and import into the manuscript

```bash
ros2 run dwpp_test_simulation dwvp_access_experiment.py summarize \
  --session /data/dwvp_access/session01
```

`trial_metrics.csv` and `summary.json` report pending/recorded/successful counts,
position and yaw errors, completion time, and mean/sample SD by task/controller.
Position projects onto a path segment; yaw is interpolated at that same point
using wrapped angular differences. Every tracking tick remains in the log,
including ticks before command streams start. The summary reports the missing
command prefix and stale samples. Failed and stale-data runs remain in counts.
Successful runs with fresh measurements supply group tracking/time summaries.
Only the first control period has a command-startup grace interval. A longer
missing prefix excludes that run from group means while retaining its status
and per-trial metrics. Check this criterion during commissioning before freezing
the experimental profile.
No significance claims follow from these descriptive statistics.

`summary.json` also keeps raw/applied command diagnostics for every recorded
trial: speed exceedances, command-increment exceedances at the configured
30 Hz, linear/angular command jerk separately, and receive-interval statistics.
The increment test uses the nominal control period, not noisy receive-time
differences. Interpret it only after checking stream gaps; it is not a measured
actuator-acceleration certificate. Failed trials retain their diagnostics.

Controller execution time and obstacle clearance are **not inferred** from the
recorder's sampling rate or action duration. Collect controller profiling and
the surveyed obstacle geometry separately when reporting those metrics. Archive
the map, obstacle layout, robot/computer details, software revisions, and videos
with the recorded session. Copy frozen numerical data to the private manuscript
repository; do not copy raw experimental output into the public plugin package.

## Offline validation

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider tests/test_access_experiment.py
docker run --rm --network none --entrypoint bash -e ROS_DOMAIN_ID=178 \
  -v "$PWD:/src:ro" docker-hsr:latest -lc \
  'source /opt/ros/humble/setup.bash && PYTHONDONTWRITEBYTECODE=1 python3 /src/tests/ros_access_recorder_smoke.py'
```

The isolated synthetic test exercises the actual ROS action client, pose-path
transport, TF, three subscribed streams, runtime snapshots, success detection,
and summary generation without a robot connection. Its five cases cover normal
operation, delayed command streams, stale odometry source times, execution timeout,
and delayed goal acceptance. It is not a controller test.

The separate `tests/ros_access_controller_smoke.py` starts the actual Humble
controller server and velocity smoother against synthetic TF, odometry, a free
map, and a kinematic plant. It checks four tangent-heading goals and two
independent-heading goals. All six completed successfully with the initial
profile. MPPI and DWVP produced lateral velocity and changed heading before the
goal on the independent-heading reference. This is an integration check, not
the paper's tracking benchmark or physical HSR evidence.

Build DWPP and DWVP into an isolated Humble workspace first, then set
`DWVP_VERIFY_WORKSPACE` to that workspace. The test requires a network-isolated
container and refuses to run with non-loopback network interfaces:

```bash
docker run --rm --network none --entrypoint bash \
  -e ROS_LOCALHOST_ONLY=1 -e ROS_DOMAIN_ID=143 -e ROS_HOME=/tmp/ros \
  -v "$DWVP_VERIFY_WORKSPACE:/ws" -v "$PWD:/experiment:ro" \
  docker-hsr:latest -c \
  'source /opt/ros/humble/setup.bash && source /ws/install/setup.bash && python3 /experiment/tests/ros_access_controller_smoke.py --output /ws/access-smoke'
```

The script exercises controller and smoother processes directly. The dedicated
map/AMCL launch, sensors, physical footprint, and the HSR driver still require a
separate commissioning check on the platform. Humble MPPI's CostCritic uses
`consider_footprint=false` with the shared circular `robot_radius` costmap.

`tests/ros_access_end_to_end_smoke.py` uses the same isolated stack and invokes
the real recorder for the constant-heading corner. Replace the test script in
the preceding command and use a fresh output directory for each run. The
verified run passed strict parameter matching for all four controllers and
produced one valid synthetic success with a 27.9 ms command-startup prefix.
All 330 pose samples remained in the log, including the initial missing-command
tick. This single timing observation is not a latency guarantee for the HSR.
