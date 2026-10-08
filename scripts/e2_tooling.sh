#!/usr/bin/env bash
# Every ROS/build/test operation uses an isolated container. No robot network.
set -euo pipefail
package_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
third_party_dir=$(dirname -- "$package_dir")
hsr_src_dir=$(dirname -- "$third_party_dir")
work=${DWVP_E2_WORK:?Set DWVP_E2_WORK to an absolute directory under /tmp}
mkdir -p "$work"
work=$(realpath "$work")
[[ "$work" == /tmp/* ]] || { echo 'Build and raw test scratch must be under /tmp' >&2; exit 2; }
input=$(realpath "${DWVP_E2_INPUT:-$work}")
image_name=${DWVP_HUMBLE_IMAGE:-docker-hsr:latest}
command=${1:?Use build, path, check-path, sweep, verify, test, or path-test}
shift
mkdir -p "$work/ws"
if [[ "$command" == test ]]; then
  simulator_dir=${DWVP_SIMULATOR_ROOT:-$(cd "$package_dir/../../../../../omnidirectional_dwvp" && pwd)}
  uv_binary=${DWVP_UV_BINARY:-$(command -v uv)}
  python_dir=$(dirname "$(dirname "$(readlink -f "$simulator_dir/.venv/bin/python")")")
  docker run --rm --network none --user "$(id -u):$(id -g)" --entrypoint bash \
    -e PYTHONDONTWRITEBYTECODE=1 -e UV_CACHE_DIR=/tmp/uv-cache \
    -v "$uv_binary:/usr/local/bin/uv:ro" -v "$python_dir:$python_dir:ro" \
    -v "$simulator_dir:$simulator_dir:ro" -v "$package_dir:/tool:ro" \
    -v "$work:/work" -w "$simulator_dir" "$image_name" -c \
    'uv run --offline --locked --python 3.11.11 --no-sync /tool/tests/generate_metrics_reference.py /work/reference'
fi
docker run --rm --network none --user "$(id -u):$(id -g)" --entrypoint bash \
  -e ROS_LOCALHOST_ONLY=1 -e ROS_DOMAIN_ID=178 -e ROS_HOME=/tmp/ros \
  -e PYTHONDONTWRITEBYTECODE=1 -e MPLCONFIGDIR=/tmp/matplotlib \
  -e DWVP_METRICS_REFERENCE=/work/reference \
  -v "$work:/work" -v "$work/ws:/ws" -v "$input:/input:ro" \
  -v "$package_dir:/ws/src/dwpp_test_simulation:ro" \
  -v "$third_party_dir/nav2_dynamic_window_pure_pursuit_controller:/ws/src/nav2_dynamic_window_pure_pursuit_controller:ro" \
  -v "$third_party_dir/nav2_omnidirectional_dwvp_controller:/ws/src/nav2_omnidirectional_dwvp_controller:ro" \
  -v "$hsr_src_dir/ytlab2_hsr_modules:/ws/src/ytlab2_hsr_modules:ro" \
  -w /ws "$image_name" -c '
    set -eo pipefail
    source /opt/ros/humble/setup.bash
    command=$1; shift
    tool=/ws/src/dwpp_test_simulation
    if [[ "$command" == build ]]; then
      colcon build --packages-select nav2_dynamic_window_pure_pursuit_controller \
        nav2_omnidirectional_dwvp_controller ytlab2_hsr_modules dwpp_test_simulation \
        --cmake-args -DBUILD_TESTING=OFF
      exit
    fi
    source /ws/install/setup.bash
    case "$command" in
      path) python3 "$tool/scripts/dwvp_access_path.py" "$@" ;;
      check-path) python3 "$tool/scripts/dwvp_access_path.py" check-path "$@" ;;
      sweep|verify) python3 "$tool/scripts/dwvp_access_e2.py" "$command" "$@" ;;
      test) python3 -m pytest -q -p no:cacheprovider "$tool"/tests/test_access_*.py ;;
      path-test) python3 "$tool/tests/ros_access_path_smoke.py" "$@" ;;
      *) echo "Unknown command: $command" >&2; exit 2 ;;
    esac
  ' bash "$command" "$@"
