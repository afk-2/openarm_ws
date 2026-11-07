"""
Make the simulated D435 depth look like a real one (simulation only).

Subscribes to Gazebo's perfect depth image and republishes it with
D435-style errors (see depth_noise.py) on a separate topic, in the real
driver's format (16UC1, millimetres). Point the estimator at that topic:

    ros2 launch cup_perception cup_pose_estimator.launch.py depth_noise:=true

Switch it off and on while running (it then passes the image through
unchanged):

    ros2 param set /d435_depth_noise enabled false

Topics
  in:  input_topic  (sensor_msgs/Image, 32FC1 metres or 16UC1 mm)
       camera_info_topic
  out: output_topic (sensor_msgs/Image, 16UC1 mm; or the input as-is when disabled)
"""
import time

import numpy as np
import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from cup_perception.cloud_ops import depth_to_metres
from cup_perception.depth_noise import add_d435_noise, NoiseSettings, to_uint16_mm


class D435DepthNoise(Node):

    def __init__(self):
        super().__init__('d435_depth_noise')
        p = self.declare_parameter
        base = '/camera/camera/aligned_depth_to_color/'
        self.enabled = p('enabled', True).value
        defaults = NoiseSettings()
        self.settings = NoiseSettings(**{n: p(n, getattr(defaults, n)).value
                                         for n in NoiseSettings.names()})
        self.rng = np.random.default_rng(p('seed', 0).value)
        self.min_period = 1.0 / max(p('max_rate', 5.0).value, 1e-3)
        self.last_stamp = None
        self.intrinsics = None
        self.add_on_set_parameters_callback(self.on_params)
        self.pub = self.create_publisher(Image, p('output_topic', base + 'image_noisy').value,
                                         qos_profile_sensor_data)
        self.create_subscription(CameraInfo, p('camera_info_topic', base + 'camera_info').value,
                                 self.on_info, qos_profile_sensor_data)
        self.create_subscription(Image, p('input_topic', base + 'image_raw').value,
                                 self.on_depth, qos_profile_sensor_data)
        self.get_logger().info(f'D435 depth noise {"ON" if self.enabled else "OFF"}: '
                               f'{self.settings}')

    def on_params(self, params):
        for prm in params:
            if prm.name == 'enabled':
                self.enabled = bool(prm.value)
                self.get_logger().info(f'D435 depth noise {"ON" if self.enabled else "OFF"}')
            elif prm.name in NoiseSettings.names():
                setattr(self.settings, prm.name, prm.value)
        return SetParametersResult(successful=True)

    def on_info(self, msg):
        self.intrinsics = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])

    def on_depth(self, msg):
        if not self.enabled:
            self.pub.publish(msg)
            return
        if self.intrinsics is None:
            return
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_stamp is not None and 0 <= stamp - self.last_stamp < self.min_period:
            return  # skip frames: each one costs ~0.2 s of CPU
        self.last_stamp = stamp
        t0 = time.monotonic()
        z = depth_to_metres(msg.data, msg.encoding, msg.height, msg.width, msg.step)
        noisy, _ = add_d435_noise(z, *self.intrinsics, self.settings, self.rng)
        out = Image(header=msg.header, height=msg.height, width=msg.width,
                    encoding='16UC1', is_bigendian=0, step=msg.width * 2)
        out.data = to_uint16_mm(noisy).tobytes()
        self.pub.publish(out)
        self.get_logger().debug(f'noise added in {(time.monotonic() - t0) * 1000:.0f} ms')


def main():
    rclpy.init()
    node = D435DepthNoise()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
