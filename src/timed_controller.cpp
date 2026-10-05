// Copyright (c) 2026 Fumiya Ohnishi
// Licensed under the MIT License.
#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "nav2_core/controller.hpp"
#include "nav2_util/node_utils.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace dwpp_test_simulation
{
// Delegation preserves controller names, tuning, exceptions and lifecycle calls.
class TimedController : public nav2_core::Controller
{
public:
  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent, std::string name,
    std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap) override
  {
    auto node = parent.lock();
    if (!node) {throw std::runtime_error("TimedController parent expired");}
    name_ = name;
    clock_ = node->get_clock();
    nav2_util::declare_parameter_if_not_declared(
      node.get(), name + ".wrapped_plugin", rclcpp::ParameterValue(std::string("")));
    const auto type = node->get_parameter(name + ".wrapped_plugin").as_string();
    if (type.empty() || type == "dwpp_test_simulation::TimedController") {
      throw std::invalid_argument("wrapped_plugin must name an actual controller");
    }
    publisher_ = node->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
      "/dwvp_access/controller_timing", rclcpp::QoS(1000).reliable());
    inner_ = loader_.createSharedInstance(type);
    inner_->configure(parent, name, std::move(tf), std::move(costmap));
  }
  void activate() override {publisher_->on_activate(); inner_->activate();}
  void deactivate() override {inner_->deactivate(); publisher_->on_deactivate();}
  void cleanup() override {inner_->cleanup(); inner_.reset(); publisher_.reset();}
  void setPlan(const nav_msgs::msg::Path & path) override {inner_->setPlan(path);}
  void setSpeedLimit(const double & limit, const bool & percentage) override
  {inner_->setSpeedLimit(limit, percentage);}
  geometry_msgs::msg::TwistStamped computeVelocityCommands(
    const geometry_msgs::msg::PoseStamped & pose, const geometry_msgs::msg::Twist & speed,
    nav2_core::GoalChecker * checker) override
  {
    const auto stamp = clock_->now();
    const auto start = std::chrono::steady_clock::now();
    geometry_msgs::msg::TwistStamped command;
    try {
      command = inner_->computeVelocityCommands(pose, speed, checker);
    } catch (...) {
      const auto stop = std::chrono::steady_clock::now();
      publish(stamp, start, stop, false);
      throw;
    }
    const auto stop = std::chrono::steady_clock::now();
    publish(stamp, start, stop, true);
    return command;
  }

private:
  void publish(
    const rclcpp::Time & stamp, std::chrono::steady_clock::time_point start,
    std::chrono::steady_clock::time_point stop, bool success)
  {
    const auto sequence = ++sequence_;
    // Instrumentation must not turn a valid command into a controller failure.
    try {
      diagnostic_msgs::msg::DiagnosticArray msg;
      msg.header.stamp = stamp;
      diagnostic_msgs::msg::DiagnosticStatus status;
      status.name = name_;
      status.hardware_id = "computeVelocityCommands";
      status.level = success ? status.OK : status.ERROR;
      status.message = success ? "returned" : "exception";
      const auto add = [&status](const std::string & key, const std::string & value) {
          diagnostic_msgs::msg::KeyValue item;
          item.key = key; item.value = value; status.values.push_back(item);
        };
      add("sequence", std::to_string(sequence));
      add("duration_ns", std::to_string(
        std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count()));
      add("success", success ? "true" : "false");
      msg.status.push_back(status);
      publisher_->publish(msg);
    } catch (...) {
      // Missing samples remain observable through sequence gaps / raw counts.
    }
  }
  pluginlib::ClassLoader<nav2_core::Controller> loader_{"nav2_core", "nav2_core::Controller"};
  nav2_core::Controller::Ptr inner_;
  rclcpp::Clock::SharedPtr clock_;
  rclcpp_lifecycle::LifecyclePublisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr publisher_;
  std::string name_;
  uint64_t sequence_{0};
};
}  // namespace dwpp_test_simulation
PLUGINLIB_EXPORT_CLASS(dwpp_test_simulation::TimedController, nav2_core::Controller)
