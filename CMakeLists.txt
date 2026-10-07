cmake_minimum_required(VERSION 3.20)
project(dwpp_test_simulation)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(nav2_core REQUIRED)
find_package(nav2_util REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(diagnostic_msgs REQUIRED)
add_library(dwvp_timed_controller SHARED src/timed_controller.cpp)
target_compile_features(dwvp_timed_controller PUBLIC cxx_std_17)
ament_target_dependencies(dwvp_timed_controller nav2_core nav2_util pluginlib
  rclcpp rclcpp_lifecycle diagnostic_msgs)
pluginlib_export_plugin_description_file(nav2_core timed_controller.xml)
install(TARGETS dwvp_timed_controller LIBRARY DESTINATION lib)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

install(DIRECTORY launch config urdf rviz params world
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/follow_path_test_gui.py
  scripts/follow_path_test_gui_real.py
  scripts/dwvp_access_experiment.py
  scripts/dwvp_access_path.py
  scripts/dwvp_access_batch.py
  scripts/dwvp_access_workflow.py
  DESTINATION lib/${PROJECT_NAME}
)

install(FILES scripts/dwvp_access_metrics.py scripts/dwvp_access_noise.py
  scripts/dwvp_access_runtime.py scripts/dwvp_access_robot_teleop.py DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
