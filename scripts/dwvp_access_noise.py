"""Passive stationary localization recording; never publishes motion commands."""
import csv
import json
import math
import time
from pathlib import Path

import numpy as np


def noise_metrics(poses):
    """Sample SD and excursion about the mean; yaw unwrapped before centering."""
    poses = np.asarray(poses, dtype=float).reshape((-1, 3))
    poses = poses[np.isfinite(poses).all(axis=1)]
    out = dict(samples=len(poses), position_std_m=None, position_max_excursion_m=None,
               x_std_m=None, y_std_m=None, x_peak_to_peak_m=None, y_peak_to_peak_m=None,
               heading_std_deg=None, heading_max_excursion_deg=None, heading_peak_to_peak_deg=None)
    if not len(poses):
        return out
    centred = poses[:, :2] - poses[:, :2].mean(axis=0)
    yaw = np.rad2deg(np.unwrap(poses[:, 2]))
    out.update(position_max_excursion_m=float(np.linalg.norm(centred, axis=1).max()),
               x_peak_to_peak_m=float(np.ptp(poses[:, 0])), y_peak_to_peak_m=float(np.ptp(poses[:, 1])),
               heading_max_excursion_deg=float(np.abs(yaw-yaw.mean()).max()),
               heading_peak_to_peak_deg=float(np.ptp(yaw)))
    if len(poses) > 1:
        std = poses[:, :2].std(axis=0, ddof=1)
        out.update(position_std_m=float(np.linalg.norm(std)), x_std_m=float(std[0]), y_std_m=float(std[1]),
                   heading_std_deg=float(yaw.std(ddof=1)))
    return out


def record_noise(output, duration=30., frequency=30., frame='map', base_frame='base_link',
                 odom_topic='/omni_base_controller/wheel_odom', max_age=.2):
    from dwvp_access_experiment import CLOCK_FUTURE_TOLERANCE_S, source_age_is_fresh
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from tf2_ros import Buffer, TransformListener

    if not all(math.isfinite(v) and v > 0 for v in (duration, frequency, max_age)):
        raise ValueError('Duration, frequency and maximum age must be positive and finite')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rclpy.init()
    node = Node('dwvp_access_stationary_noise')
    buffer = Buffer()
    listener = TransformListener(buffer, node)
    latest = []
    sub = node.create_subscription(Odometry, odom_topic,
        lambda m: latest.__setitem__(slice(None), [m, time.monotonic()]), qos_profile_sensor_data)
    samples, rejected, moving, duplicate = [], 0, 0, 0
    started = time.monotonic()
    tick = started
    previous_stamp = None
    try:
        with (output/'noise_samples.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['t','stamp_s','tf_stamp_s','x','y','yaw','tf_age_s','odom_age_s',
                             'odom_source_age_s','speed_m_s','yaw_rate_rad_s','accepted'])
            while time.monotonic()-started < duration:
                rclpy.spin_once(node, timeout_sec=.005)
                mono = time.monotonic()
                if mono < tick:
                    continue
                tick = mono+1/frequency
                pose = [math.nan]*3
                tf_stamp = math.nan
                tf_age = odom_age = source_age = math.inf
                speed = yaw_rate = math.nan
                try:
                    tf = buffer.lookup_transform(frame, base_frame, rclpy.time.Time())
                    q = tf.transform.rotation
                    pose = [tf.transform.translation.x, tf.transform.translation.y,
                            math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))]
                    tf_stamp = tf.header.stamp.sec+tf.header.stamp.nanosec/1e9
                except Exception:
                    pass
                now = node.get_clock().now().nanoseconds/1e9
                tf_age = now-tf_stamp
                if latest:
                    odom, received = latest
                    odom_age = mono-received
                    source_age = now-(odom.header.stamp.sec+odom.header.stamp.nanosec/1e9)
                    v = odom.twist.twist
                    speed, yaw_rate = math.hypot(v.linear.x, v.linear.y), abs(v.angular.z)
                fresh = (np.isfinite(pose).all() and 0 <= odom_age <= max_age
                         and source_age_is_fresh(tf_age, max_age) and source_age_is_fresh(source_age, max_age)
                         and math.isfinite(speed) and math.isfinite(yaw_rate))
                stationary = fresh and speed <= .005 and yaw_rate <= .01
                moving += int(fresh and not stationary)
                repeated = tf_stamp == previous_stamp
                duplicate += int(stationary and repeated)
                accepted = stationary and not repeated
                if accepted:
                    samples.append(pose)
                    previous_stamp = tf_stamp
                rejected += int(not stationary)
                writer.writerow([mono-started, now, tf_stamp, *pose, tf_age, odom_age, source_age,
                                 speed, yaw_rate, accepted])
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    result = noise_metrics(samples)
    result.update(duration_s=time.monotonic()-started, requested_duration_s=duration, frame=frame,
                  maximum_future_source_skew_s=CLOCK_FUTURE_TOLERANCE_S,
                  base_frame=base_frame, rejected_samples=rejected, duplicate_tf_samples=duplicate,
                  moving_samples=moving, stationary_verified=bool(len(samples)>1 and moving==0),
                  stationary_speed_limit_m_s=.005, stationary_yaw_rate_limit_rad_s=.01,
                  note='Passive TF/odometry observation. Position SD=sqrt(var(x)+var(y)); maximum excursion is distance from the sample mean. Sample SD uses n-1. Yaw is unwrapped. Correlated localization samples are not independent trials.')
    (output/'noise_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (output/'noise_summary.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(result)); writer.writeheader(); writer.writerow(result)
    return result
