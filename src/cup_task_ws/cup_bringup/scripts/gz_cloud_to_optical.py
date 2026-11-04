#!/usr/bin/env python3
"""
Republish Gazebo's RGB-D point cloud in optical axes.

Gazebo Fortress fills the rgbd_camera cloud in the sensor's link axes
(x forward, y left, z up) even though it stamps it with ignition_frame_id
(camera_color_optical_frame), so it shows up rotated in RViz. Rotate the
points into optical axes (x right, y down, z forward), as realsense2_camera
publishes them; the rgb field is left untouched.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2


class GzCloudToOptical(Node):

    def __init__(self):
        super().__init__('gz_cloud_to_optical')
        self.pub = self.create_publisher(PointCloud2, 'points_out', qos_profile_sensor_data)
        self.create_subscription(PointCloud2, 'points_in', self.callback, qos_profile_sensor_data)

    def callback(self, msg):
        offsets = {f.name: f.offset for f in msg.fields}
        if (offsets.get('x'), offsets.get('y'), offsets.get('z')) != (0, 4, 8) or \
                msg.point_step % 4:
            self.get_logger().error('Unexpected PointCloud2 layout', throttle_duration_sec=5.0)
            return
        data = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, msg.point_step // 4).copy()
        x, y, z = data[:, 0].copy(), data[:, 1].copy(), data[:, 2].copy()
        data[:, 0], data[:, 1], data[:, 2] = -y, -z, x
        msg.data = data.tobytes()
        self.pub.publish(msg)


def main():
    rclpy.init()
    rclpy.spin(GzCloudToOptical())


if __name__ == '__main__':
    main()
