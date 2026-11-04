"""
Cup pose source: Gazebo ground truth (SIMULATION ONLY, stand-in for vision).

Publishes the cup's true pose on /cup_pose (geometry_msgs/PoseStamped) in the
robot's "world" frame. The pick-and-place task only listens to /cup_pose, so
a perception node can replace this one later by publishing the same topic
with the same convention:

    position     centre of the cup's base (where it touches the table)
    orientation  cup z axis up, x axis pointing towards the handle

Input: /gazebo/dynamic_poses (tf2_msgs/TFMessage), Gazebo's world poses of all
moving models, bridged by cup_bringup/config/gz_bridge.yaml.

Parameters
----------
model_name  Gazebo model to track (default: ceramic_cup)
spawn_xyz   robot spawn position in the Gazebo world; the robot's "world"
            frame is the robot base, so poses are shifted by this
frame_id    output frame (default: world)
rate        maximum publish rate in Hz (default: 10)

"""
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from tf2_msgs.msg import TFMessage


class GazeboCupPose(Node):

    def __init__(self):
        super().__init__('gazebo_cup_pose')
        self.model_name = self.declare_parameter('model_name', 'ceramic_cup').value
        self.spawn_xyz = self.declare_parameter('spawn_xyz', [0.0, 0.0, 0.40]).value
        self.frame_id = self.declare_parameter('frame_id', 'world').value
        self.period = 1.0 / self.declare_parameter('rate', 10.0).value
        self.last_stamp = None
        self.pub = self.create_publisher(PoseStamped, 'cup_pose', 10)
        self.create_subscription(TFMessage, '/gazebo/dynamic_poses', self.callback,
                                 QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.get_logger().info(
            f'Publishing Gazebo ground truth of "{self.model_name}" on /cup_pose '
            '(simulation-only stand-in for perception)')

    def callback(self, msg):
        now = self.get_clock().now()
        if self.last_stamp is not None and (now - self.last_stamp).nanoseconds < self.period * 1e9:
            return
        for tf in msg.transforms:
            if tf.child_frame_id != self.model_name:
                continue
            out = PoseStamped()
            out.header.stamp = now.to_msg()
            out.header.frame_id = self.frame_id
            t = tf.transform.translation
            out.pose.position.x = t.x - self.spawn_xyz[0]
            out.pose.position.y = t.y - self.spawn_xyz[1]
            out.pose.position.z = t.z - self.spawn_xyz[2]
            out.pose.orientation = tf.transform.rotation
            self.pub.publish(out)
            self.last_stamp = now
            return


def main():
    rclpy.init()
    rclpy.spin(GazeboCupPose())


if __name__ == '__main__':
    main()
