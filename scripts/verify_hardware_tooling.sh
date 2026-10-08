#!/usr/bin/env bash
# Network-isolated synthetic checks only. Never uses the HSR Compose network.
set -euo pipefail
package_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
third_party_dir=$(dirname -- "$package_dir")
hsr_src_dir=$(dirname -- "$third_party_dir")
verify_root=${DWVP_ACCESS_VERIFY_ROOT:-$(mktemp -d /tmp/dwvp-access-verify.XXXXXX)}
mkdir -p "$verify_root"
run_dir=$(mktemp -d "$verify_root/run.XXXXXX")
image_name=${DWVP_HUMBLE_IMAGE:-docker-hsr:latest}
printf 'Synthetic verification artifacts: %s\n' "$run_dir"
simulator_dir=${DWVP_SIMULATOR_ROOT:-$(cd "$package_dir/../../../../../omnidirectional_dwvp" && pwd)}
uv_binary=${DWVP_UV_BINARY:-$(command -v uv)}
python_dir=$(dirname "$(dirname "$(readlink -f "$simulator_dir/.venv/bin/python")")")
# Reuse installed, locked dependencies. All simulator and interpreter mounts are
# read-only; --no-sync prevents uv from modifying the existing environment.
docker run --rm --network none --user "$(id -u):$(id -g)" --entrypoint bash \
  -e PYTHONDONTWRITEBYTECODE=1 -e UV_CACHE_DIR=/tmp/uv-cache \
  -v "$uv_binary:/usr/local/bin/uv:ro" -v "$python_dir:$python_dir:ro" \
  -v "$simulator_dir:$simulator_dir:ro" -v "$package_dir:/tool:ro" \
  -v "$run_dir:/output" -w "$simulator_dir" "$image_name" -c '
    uv run --offline --locked --python 3.11.11 --no-sync /tool/tests/generate_metrics_reference.py /output/reference
  '
docker run --rm --network none --user "$(id -u):$(id -g)" \
  --entrypoint bash -e ROS_LOCALHOST_ONLY=1 -e ROS_DOMAIN_ID=178 \
  -e ROS_HOME=/tmp/ros -e PYTHONDONTWRITEBYTECODE=1 -e MPLCONFIGDIR=/tmp/matplotlib \
  -e DWVP_METRICS_REFERENCE=/ws/reference \
  -v "$run_dir:/ws" \
  -v "$package_dir:/ws/src/dwpp_test_simulation:ro" \
  -v "$third_party_dir/nav2_dynamic_window_pure_pursuit_controller:/ws/src/nav2_dynamic_window_pure_pursuit_controller:ro" \
  -v "$third_party_dir/nav2_omnidirectional_dwvp_controller:/ws/src/nav2_omnidirectional_dwvp_controller:ro" \
  -v "$hsr_src_dir/ytlab2_hsr_modules:/ws/src/ytlab2_hsr_modules:ro" \
  -v "$third_party_dir/tmc/hsrb_rosnav/hsrb_mapping:/ws/src/hsrb_mapping:ro" \
  -w /ws "$image_name" -c '
    set -eo pipefail
    source /opt/ros/humble/setup.bash
    colcon build --packages-select nav2_dynamic_window_pure_pursuit_controller \
      nav2_omnidirectional_dwvp_controller hsrb_mapping ytlab2_hsr_modules dwpp_test_simulation \
      --cmake-args -DBUILD_TESTING=OFF
    source /ws/install/setup.bash
    python3 -m pytest -q -p no:cacheprovider /ws/src/dwpp_test_simulation/tests/test_access_*.py
    python3 /ws/src/dwpp_test_simulation/tests/ros_access_recorder_smoke.py --output /ws/recorder-smoke > /ws/recorder-smoke.log 2>&1 || {
      cat /ws/recorder-smoke.log
      exit 1
    }
    python3 /ws/src/dwpp_test_simulation/tests/ros_access_controller_smoke.py --output /ws/controller-smoke
    python3 /ws/src/dwpp_test_simulation/tests/ros_access_end_to_end_smoke.py --output /ws/end-to-end-smoke
    python3 /ws/src/dwpp_test_simulation/tests/ros_access_batch_smoke.py --bidirectional --all-methods --output /ws/batch-smoke
    python3 /ws/src/dwpp_test_simulation/tests/ros_access_current_start_smoke.py --output /ws/current-start-smoke
    python3 /ws/src/dwpp_test_simulation/tests/ros_access_endpoint_resume_smoke.py --output /ws/endpoint-resume-smoke
    python3 /ws/src/dwpp_test_simulation/tests/ros_access_failed_retry_smoke.py --inject-turn-drift --output /ws/failed-retry-smoke
    python3 /ws/src/dwpp_test_simulation/tests/ros_access_failed_retry_smoke.py --reposition-before-resume --output /ws/repositioned-retry-smoke
    python3 /ws/src/dwpp_test_simulation/tests/ros_access_path_smoke.py --output /ws/path-smoke
    python3 /ws/src/dwpp_test_simulation/tests/ros_access_workflow_smoke.py --output /ws/workflow-smoke
    ros2 launch dwpp_test_simulation dwvp_access_hsr.launch.py --show-args
  '
