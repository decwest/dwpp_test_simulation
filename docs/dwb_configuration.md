# DWB configuration for the 65-trial HSR plan

DWB is assigned only to `E2_environment` (five planned repeats). This is a starting
configuration for author review, not a physically tuned or validated HSR profile.
Physical HSR trials remain zero.

The baseline is the DWB `FollowPath` block in the installed Humble image's
`/opt/ros/humble/share/nav2_bringup/params/nav2_params.yaml`. Its upstream counterpart
is [Humble nav2_params.yaml](https://github.com/ros-navigation/navigation2/blob/humble/nav2_bringup/params/nav2_params.yaml).
Code defaults and bringup example values differ; the table uses the **bringup
example**, with absent values explicitly marked. The actual planner remains
`dwb_core::DWBLocalPlanner` under `dwpp_test_simulation::TimedController`.

| Setting | Humble bringup baseline | Experiment value | Reason |
|---|---|---|---|
| plugin | DWB directly | TimedController wrapping DWB | Same per-call measurement as the other six methods |
| trajectory_generator_name | absent; code defaults to StandardTrajectoryGenerator | dwb_plugins::LimitedAccelGenerator | Sample the one-cycle reachable dynamic window |
| sim_period | absent | 1/30 s | Match the common controller period |
| min_vel_x, max_vel_x | 0, 0.26 | −0.22, 0.22 m/s | Shared body-x box, including reverse |
| min_vel_y, max_vel_y | 0, 0 | −0.22, 0.22 m/s | Enable lateral motion under the shared box |
| max_vel_theta | 1.0 | 0.6 rad/s | Shared angular velocity bound |
| max_speed_xy | 0.26 | hypot(0.22,0.22) = 0.311127 m/s | Avoid an additional norm constraint inside the shared component box |
| acc_lim_x, acc_lim_y, acc_lim_theta | 2.5, 0, 3.2 | 0.22, 0.22, 0.6 | Shared acceleration magnitudes |
| decel_lim_x, decel_lim_y, decel_lim_theta | −2.5, 0, −3.2 | −0.22, −0.22, −0.6 | Shared deceleration magnitudes |
| xy_goal_tolerance | 0.25 | 0.1 m | Align RotateToGoal with the common goal checker |
| trans_stopped_velocity | 0.25 | 0.005 m/s | Enter rotation only once zero translation is reachable within a nominal 0.22/30 m/s step |
| stateful | true in the DWB example block | omitted | Humble DWB does not declare this unused entry; the shared goal checker is explicitly non-stateful |
| sim_time | 1.7 | 8.0 s | Resolve forward progress on the 0.05 m costmap under the narrow startup dynamic window |
| Oscillation.oscillation_reset_time | absent; code default −1 (disabled) | 1.0 s | Release a direction restriction even when the low-speed terminal motion cannot meet the distance/angle reset thresholds |

The [Humble LimitedAccelGenerator implementation](https://github.com/ros-navigation/navigation2/blob/humble/nav2_dwb_controller/dwb_plugins/src/limited_accel_generator.cpp)
limits candidate velocities using `sim_period`; the default StandardTrajectoryGenerator
uses the trajectory simulation interval instead. No upstream code is modified.
Acceleration/deceleration rendering follows each condition's scale. DWB is assigned
the nominal scale; a rendered half profile also uses half bounds consistently,
although DWB is not assigned to an orientation experiment.

The 1.7 s starting value stalled at the initial pose in the synthetic tangent-path
check: a maximum startup axial command of 0.22/30 m/s predicts only 0.01247 m,
smaller than a 0.05 m grid cell, so grid critic scores can tie. At 8 s it predicts
0.05867 m. The isolated diagnostic then completed the same route. This longer
horizon is a documented adaptation to the low acceleration setting; it requires
author review on real maps and is not evidence of optimal DWB tuning. At higher
speeds it may reject rollouts leaving the existing local costmap, reducing speed.
The 0.005 m/s stop threshold also avoids entering the pure-rotation phase while
zero translation is outside the next reachable window. Critic weights are unchanged.

With the default time reset disabled, a terminal run issued tiny angular commands
in the wrong direction until the progress checker aborted. The
[Humble Oscillation critic](https://github.com/ros-navigation/navigation2/blob/humble/nav2_dwb_controller/dwb_critics/src/oscillation.cpp)
can forbid a sign reversal after a previous reversal, until motion exceeds its
reset distance or angle. Setting the time reset to 1 s also releases this
restriction when terminal motion is too small to reach those thresholds. This
retains the critic and its distance/angle thresholds; it requires author review
along with the longer horizon. RotateToGoal's lookahead remains unchanged.

The following bringup values are retained: `vx_samples=20`, `vy_samples=5`,
`vtheta_samples=20`, `linear_granularity=0.05`,
`angular_granularity=0.025`, `transform_tolerance=0.2`, `min_speed_xy=0`,
`min_speed_theta=0`, `debug_trajectory_details=true`,
`short_circuit_trajectory_evaluation=true`.
The existing five y samples become active because y velocity and acceleration
limits are nonzero. No sample-count or critic-weight tuning was performed.

The seven critics are unchanged: RotateToGoal, Oscillation, BaseObstacle,
GoalAlign, PathAlign, PathDist and GoalDist. Their configured scales remain,
respectively, 32, the code default, 0.02, 24, 32, 32 and 24. PathAlign/GoalAlign
forward distances remain 0.1 m; RotateToGoal slowing factor remains 5 and
lookahead time remains −1 s. Other omitted DWB/critic settings retain their
installed Humble defaults.

The shared server differs from the bringup example in `controller_frequency`
(20→30 Hz), `min_y_velocity_threshold` (0.5→0.001 m/s), progress radius/time
(0.5 m/10 s→0.1 m/30 s), and goal checker settings (stateful true→false,
xy 0.25→0.1 m, yaw 0.25→0.3 rad). These are the existing common experiment
settings, now also used by DWB. The common goal checker controls completion for
all seven methods. The unused `DWB.stateful` entry was removed after the runtime
parameter snapshot exposed that the planner does not declare it. Parameter
verification still requires every configured field of the selected controller.
