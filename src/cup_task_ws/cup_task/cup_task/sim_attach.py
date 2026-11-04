"""
SIMULATION-ONLY SHORTCUT: lock the cup to the hand in Gazebo while it is held.

Gazebo models finger contacts as single points with no grip area, so a cup
pinched between two fingertips pivots about the line between them and tilts
(about 10 deg in testing). Real finger pads resist that rotation. This helper
drives Gazebo's DetachableJoint plugin, which cup_description adds to the
robot with `sim.launch.py sim_cup_attach:=true`:

    attach  /cup_sim_attach/attach   (after the gripper has closed)
    detach  /cup_sim_attach/detach   (before the gripper opens)

Never use it on the real robot (`use_sim_attach:=false`).
"""
import time

from std_msgs.msg import Empty


class SimAttach:

    def __init__(self, node):
        self.node = node
        self.attach_pub = node.create_publisher(Empty, '/cup_sim_attach/attach', 10)
        self.detach_pub = node.create_publisher(Empty, '/cup_sim_attach/detach', 10)
        node.get_logger().warn('use_sim_attach is on: the cup will be locked to the hand in '
                               'Gazebo while held (simulation-only shortcut)')

    def _send(self, pub, what):
        # Publish a few times: the bridge may only just have matched the publisher.
        for _ in range(3):
            pub.publish(Empty())
            time.sleep(0.2)
        self.node.get_logger().info(f'       [sim only] cup {what} the hand in Gazebo')

    def attach(self):
        self._send(self.attach_pub, 'locked to')

    def detach(self):
        self._send(self.detach_pub, 'released from')
