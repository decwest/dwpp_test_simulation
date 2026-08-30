# Nav2 MPPI local patch

`nav2_mppi_last_command_velocity_1.3.12.patch` is based on the official
Navigation2 `1.3.12` tag:

- Tag commit: `6be3614013ec586051b86c97b919b293281490fe`
- Package: `nav2_mppi_controller`
- Local parameter: `use_last_command_velocity`

The patch does not contain the upstream `1.3.10 -> 1.3.12` changes. Start from
the official `1.3.12` package, then apply it from the Navigation2 source root:

```bash
patch -p1 < /home/ubuntu/ros2_ws/src/dwpp_test_simulation/patches/nav2_mppi_last_command_velocity_1.3.12.patch
```

Build and test:

```bash
source /opt/ros/jazzy/setup.bash
cd /home/ubuntu/ros2_ws
colcon build \
  --symlink-install \
  --packages-select nav2_mppi_controller \
  --cmake-args -DBUILD_TESTING=ON
source install/setup.bash
colcon test --packages-select nav2_mppi_controller
colcon test-result --test-result-base build/nav2_mppi_controller --verbose
ros2 pkg prefix nav2_mppi_controller
```

The final command must resolve to the workspace overlay, not `/opt/ros/jazzy`.

With the option enabled, MPPI matches DWPP's velocity-state update:

- the initial stored command is zero;
- each successful raw controller output becomes the next rollout's initial velocity;
- `reset()` and `deactivate()` clear the stored command;
- `setPlan()` and computation exceptions preserve it;
- downstream velocity-smoother and collision-monitor outputs are not fed back.

