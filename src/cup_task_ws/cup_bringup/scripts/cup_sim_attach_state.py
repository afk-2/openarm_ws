#!/usr/bin/env python3
"""
SIMULATION ONLY: free the cup at startup, and keep track of the cup lock.

Gazebo's DetachableJoint (sim.launch.py sim_cup_attach:=true) locks the cup
to the hand as soon as the robot loads, wherever the cup is: on the table,
so the cup would then float along with the hand. This node sends "detach"
until Gazebo confirms it, instead of a few messages that are lost when
Gazebo is slow to start listening (seen with the Gazebo GUI on WSL).

Gazebo reports the lock state only when it changes ("attached"/"detached"
on /cup_sim_attach/state, bridged). This node republishes it latched on
/cup_sim_attach/attached (std_msgs/Bool, transient local), so cup_task can
always read the current state.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Bool, Empty, String

RETRY_PERIOD = 0.5   # s (wall time) between detach requests at startup
WARN_PERIOD = 10.0   # s between "still locked" warnings


class CupSimAttachState(Node):

    def __init__(self):
        super().__init__('cup_sim_attach_state')
        self.attached = None
        self.released = False
        self.waited = 0.0
        self.detach_pub = self.create_publisher(Empty, '/cup_sim_attach/detach', 10)
        self.state_pub = self.create_publisher(
            Bool, '/cup_sim_attach/attached',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(String, '/cup_sim_attach/state', self.on_state, 10)
        self.timer = self.create_timer(RETRY_PERIOD, self.release)

    def on_state(self, msg):
        if msg.data not in ('attached', 'detached'):
            return
        self.attached = msg.data == 'attached'
        self.state_pub.publish(Bool(data=self.attached))
        if not self.attached and not self.released:
            self.released = True
            self.timer.cancel()
            self.get_logger().info(f'cup released from the hand after {self.waited:.1f} s '
                                   '(it is locked to the hand when the robot loads)')

    def release(self):
        self.detach_pub.publish(Empty())
        self.waited += RETRY_PERIOD
        if self.waited % WARN_PERIOD < RETRY_PERIOD:
            self.get_logger().warn(f'cup still locked to the hand after {self.waited:.0f} s; '
                                   'still sending detach')


def main():
    rclpy.init()
    rclpy.spin(CupSimAttachState())


if __name__ == '__main__':
    main()
