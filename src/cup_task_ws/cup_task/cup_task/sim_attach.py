"""
SIMULATION-ONLY SHORTCUT: lock the cup to the hand in Gazebo while it is held.

Gazebo models finger contacts as single points with no grip area, so a cup
pinched between two fingertips pivots about the line between them and tilts
(about 10 deg in testing). Real finger pads resist that rotation. This helper
drives Gazebo's DetachableJoint plugin, which cup_description adds to the
robot with `sim.launch.py sim_cup_attach:=true`:

    attach  /cup_sim_attach/attach   (after the gripper has closed)
    detach  /cup_sim_attach/detach   (before the gripper opens)

Each request is repeated until Gazebo confirms it: cup_bringup's
cup_sim_attach_state.py keeps the lock state latched on
/cup_sim_attach/attached. (A few unconfirmed messages can be lost while
Gazebo is slow, and a cup that stays locked floats along with the hand.)

Never use it on the real robot (`use_sim_attach:=false`).
"""
import math
import time

from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Empty
from tf2_msgs.msg import TFMessage

from cup_task.moveit_client import TaskError

CUP_MODEL = 'ceramic_cup'  # Gazebo model name (cup_description/worlds/cup_table.sdf)
RETRY_PERIOD = 0.25   # s between requests
CONFIRM_TIMEOUT = 20.0  # s (wall time) to wait for Gazebo to confirm
STATE_WAIT = 3.0      # s to wait for the first latched state


class SimAttach:

    def __init__(self, node):
        self.node = node
        self.attach_pub = node.create_publisher(Empty, '/cup_sim_attach/attach', 10)
        self.detach_pub = node.create_publisher(Empty, '/cup_sim_attach/detach', 10)
        # The cup's true position (Gazebo ground truth), for the missed-cup
        # check of gripper_close_mode "width". The robot is spawned at x = y = 0,
        # so Gazebo's x, y are the robot's.
        self.cup_xy = None
        self.attached = None  # Gazebo's lock state; None = not known (yet)
        node.create_subscription(
            Bool, '/cup_sim_attach/attached', self._on_state,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        node.create_subscription(
            TFMessage, '/gazebo/dynamic_poses', self._on_poses,
            QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT))
        node.get_logger().warn('use_sim_attach is on: the cup will be locked to the hand in '
                               'Gazebo while held (simulation-only shortcut)')

    def _on_state(self, msg):
        self.attached = msg.data

    def _send(self, pub, want, what):
        """Request until Gazebo reports the lock state `want`."""
        log = self.node.get_logger()
        end = time.monotonic() + STATE_WAIT
        while self.attached is None and time.monotonic() < end:
            time.sleep(0.05)
        if self.attached is None:
            # No state node (sim launched without it): send a few and hope.
            log.warn('       [sim only] cup lock state unknown (is cup_sim_attach_state.py '
                     'running?); sending the request without confirmation')
            for _ in range(5):
                pub.publish(Empty())
                time.sleep(0.2)
            return
        start = time.monotonic()
        while self.attached != want:
            if time.monotonic() - start > CONFIRM_TIMEOUT:
                raise TaskError(f'[sim only] Gazebo did not confirm the cup {what} the hand '
                                f'within {CONFIRM_TIMEOUT:.0f} s')
            pub.publish(Empty())
            time.sleep(RETRY_PERIOD)
        log.info(f'       [sim only] cup {what} the hand in Gazebo '
                 f'(confirmed after {time.monotonic() - start:.1f} s)')

    def _on_poses(self, msg):
        for tf in msg.transforms:
            if tf.child_frame_id == CUP_MODEL:
                self.cup_xy = (tf.transform.translation.x, tf.transform.translation.y)

    def cup_distance(self, xy, timeout=10.0):
        """Horizontal distance (m) from `xy` to the cup's true position in Gazebo."""
        end = time.monotonic() + timeout
        while self.cup_xy is None and time.monotonic() < end:
            time.sleep(0.05)
        if self.cup_xy is None:
            raise RuntimeError('no cup pose from Gazebo on /gazebo/dynamic_poses')
        return math.hypot(self.cup_xy[0] - xy[0], self.cup_xy[1] - xy[1])

    def attach(self):
        self._send(self.attach_pub, True, 'locked to')

    def detach(self):
        self._send(self.detach_pub, False, 'released from')
