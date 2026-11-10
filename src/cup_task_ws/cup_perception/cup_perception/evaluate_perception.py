"""
Evaluate cup_pose_estimator against Gazebo's true cup poses (SIMULATION ONLY).

For each case in a YAML test set (config/perception_eval.yaml) it puts the
cups where the case says (Gazebo set_pose), optionally moves an arm with
MoveIt, switches the depth noise off and on, collects estimator frames and
compares them with the true poses. A second part turns one cup all the way
round to find which handle angles the camera cannot see.

Needs, in the same ROS domain and Gazebo partition:

    ros2 launch cup_bringup sim.launch.py gui:=false rviz:=false \\
        sim_cup_attach:=false moveit:=true        # MoveIt only for arm cases
    ros2 launch cup_perception cup_pose_estimator.launch.py depth_noise:=true
    ros2 run cup_perception evaluate_perception [--cases FILE] [--out DIR]
        [--only CASE ...] [--no-sweep] [--no-cases] [--replot]

--replot only redraws the plots from the CSV files (no simulation needed).

Writes to the output folder (default: <workspace>/src/cup_task_ws/results):
    cases.csv               one row per case, noise setting and cup
    sweep.csv               handle visibility per handle angle
    summary.txt             the readable printout
    error_vs_distance.png   position error vs distance from the camera
    handle_vs_angle.png     handle visibility and yaw error vs handle angle
"""
import argparse
import csv
import math
import os
import subprocess
import threading
import time

from ament_index_python.packages import get_package_prefix, get_package_share_directory
import numpy as np
import rclpy
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String
import tf2_ros
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import MarkerArray
import yaml

WORLD = 'cup_table'           # Gazebo world name (cup_description/worlds/cup_table.sdf)
SPAWN_XYZ = [0.0, 0.0, 0.40]  # robot spawn position = offset of the "world" frame (--spawn-z)
TABLE_Z = 0.75                # table top in Gazebo coordinates
FIRST_CUP = 'ceramic_cup'     # the cup already in the world


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


# ---- geometry: can the camera see the handle at all? ------------------------
def handle_samples(cup, n_arc=24, n_ring=12):
    """Points and outward normals on the handle's tube surface (cup frame)."""
    h = cup['handle']
    r_tube, loop = h['thickness'] / 2, h['radius']
    t = np.linspace(-math.pi / 2, math.pi / 2, n_arc)
    x0 = cup['outer_diameter'] / 2 - 0.25 * h['thickness']  # as generate_cup_model.py
    centre = np.stack([x0 + loop * np.cos(t), np.zeros_like(t),
                       h['center_height'] + loop * np.sin(t)], axis=1)
    radial = np.stack([np.cos(t), np.zeros_like(t), np.sin(t)], axis=1)
    phi = np.linspace(0, 2 * math.pi, n_ring, endpoint=False)
    normals = (np.cos(phi)[None, :, None] * radial[:, None, :]
               + np.sin(phi)[None, :, None] * np.array([0.0, 1.0, 0.0]))
    pts = centre[:, None, :] + r_tube * normals
    pts, normals = pts.reshape(-1, 3), normals.reshape(-1, 3)
    outside = np.hypot(pts[:, 0], pts[:, 1]) > cup['outer_diameter'] / 2
    return pts[outside], normals[outside]


def handle_visible_fraction(cup, x, y, base_z, yaw, camera):
    """
    Share of the handle's surface the camera can see (0..1), from geometry.

    A surface point counts if it faces the camera and the line to the camera
    does not pass through the cup body (a cylinder).
    """
    pts, normals = handle_samples(cup)
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    pts = pts @ rot.T + [x, y, base_z]
    normals = normals @ rot.T
    to_cam = camera - pts
    facing = np.einsum('ij,ij->i', normals, to_cam) > 0
    # Line p + s * to_cam (0 < s < 1) against the body cylinder.
    r = cup['outer_diameter'] / 2
    px, py = pts[:, 0] - x, pts[:, 1] - y
    dx, dy = to_cam[:, 0], to_cam[:, 1]
    a = dx * dx + dy * dy
    b = 2 * (px * dx + py * dy)
    cc = px * px + py * py - r * r
    disc = b * b - 4 * a * cc
    blocked = np.zeros(len(pts), bool)
    ok = disc > 0
    for sign in (-1, 1):
        sroot = np.full(len(pts), -1.0)
        sroot[ok] = (-b[ok] + sign * np.sqrt(disc[ok])) / (2 * a[ok])
        z = pts[:, 2] + sroot * to_cam[:, 2]
        blocked |= (sroot > 1e-6) & (sroot < 1) & (z > base_z) & (z < base_z + cup['height'])
    return float((facing & ~blocked).mean())


# ---- the ROS side ------------------------------------------------------------
class Evaluator(Node):

    def __init__(self):
        super().__init__('evaluate_perception', parameter_overrides=[
            rclpy.parameter.Parameter('use_sim_time', value=True)])
        self.frames = []      # (stamp ns, [estimate dicts])
        self.truth = {}       # Gazebo model name -> (x, y, z, yaw) in the world frame
        self.info = None
        self.lock = threading.Lock()
        self.tf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf, self)
        self.create_subscription(MarkerArray, '/cup_perception/markers', self.on_markers, 10)
        self.create_subscription(TFMessage, '/gazebo/dynamic_poses', self.on_truth,
                                 QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.create_subscription(CameraInfo, '/camera/camera/aligned_depth_to_color/camera_info',
                                 lambda m: setattr(self, 'info', m), 5)
        self.robot_pts = []   # "on robot N" from each status line
        self.create_subscription(String, '/cup_perception/status', self.on_status, 10)
        self.noise_client = self.create_client(SetParameters, '/d435_depth_noise/set_parameters')
        self.moveit = None

    def on_markers(self, msg):
        bodies = {m.id: m for m in msg.markers if m.ns == 'cup'}
        labels = {m.id: m.text for m in msg.markers if m.ns == 'label'}
        if not msg.markers:
            return
        stamp = msg.markers[0].header.stamp
        est = [{'x': b.pose.position.x, 'y': b.pose.position.y,
                'z': b.pose.position.z - b.scale.z / 2, 'yaw': yaw_of(b.pose.orientation),
                'visible': 'handle seen' in labels.get(i, '')} for i, b in bodies.items()]
        with self.lock:
            self.frames.append((stamp.sec * 10**9 + stamp.nanosec, est))
            del self.frames[:-200]

    def on_status(self, msg):
        if 'on robot ' in msg.data:
            with self.lock:
                self.robot_pts.append(int(msg.data.split('on robot ')[1].split(',')[0]))
                del self.robot_pts[:-200]

    def on_truth(self, msg):
        for t in msg.transforms:
            p, q = t.transform.translation, t.transform.rotation
            self.truth[t.child_frame_id] = (p.x - SPAWN_XYZ[0], p.y - SPAWN_XYZ[1],
                                            p.z - SPAWN_XYZ[2], yaw_of(q))

    # ---- helpers ---------------------------------------------------------------
    def now_ns(self):
        return self.get_clock().now().nanoseconds

    def sim_sleep(self, seconds):
        end = self.now_ns() + seconds * 1e9
        while self.now_ns() < end:
            time.sleep(0.05)

    def gz(self, service, reqtype, req):
        cmd = ['ign', 'service', '-s', f'/world/{WORLD}/{service}', '--reqtype', reqtype,
               '--reptype', 'ignition.msgs.Boolean', '--timeout', '5000', '--req', req]
        for _ in range(3):  # a slow simulation sometimes misses the 5 s timeout
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
            if 'data: true' in out:
                return
        raise RuntimeError(f'Gazebo {service} failed: {out.strip() or "no reply"}')

    @staticmethod
    def _pose_req(x, y, yaw):
        """Gazebo pose for a cup standing on the table at world (x, y)."""
        return (f'position: {{x: {x + SPAWN_XYZ[0]}, y: {y + SPAWN_XYZ[1]}, '
                f'z: {TABLE_Z + 0.001}}}, orientation: {{z: {math.sin(yaw / 2)}, '
                f'w: {math.cos(yaw / 2)}}}')

    def ensure_cups(self, names):
        for name in names:
            if name not in self.truth:
                self.gz('create', 'ignition.msgs.EntityFactory',
                        f'sdf_filename: "model://ceramic_cup", name: "{name}", '
                        'pose: {position: {x: -1.5, y: 2.5, z: 0.001}}')
                end = time.monotonic() + 30
                while name not in self.truth and time.monotonic() < end:
                    time.sleep(0.1)

    def place(self, name, x, y, yaw):
        self.gz('set_pose', 'ignition.msgs.Pose', f'name: "{name}", ' + self._pose_req(x, y, yaw))

    def park(self, name, i, park_xy):
        req = (f'name: "{name}", position: {{x: {park_xy[0] + 0.2 * i}, y: {park_xy[1]}, '
               f'z: 0.001}}, orientation: {{w: 1.0}}')
        self.gz('set_pose', 'ignition.msgs.Pose', req)

    def set_noise(self, on):
        if not self.noise_client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError('/d435_depth_noise not running: start the estimator with '
                               'depth_noise:=true')
        value = ParameterValue(type=ParameterType.PARAMETER_BOOL, bool_value=on)
        future = self.noise_client.call_async(SetParameters.Request(
            parameters=[Parameter(name='enabled', value=value)]))
        end = time.monotonic() + 10
        while not future.done() and time.monotonic() < end:
            time.sleep(0.05)

    def camera(self):
        """Camera position (world) and world->optical rotation."""
        frame = self.info.header.frame_id
        t = self.tf.lookup_transform('world', frame, rclpy.time.Time(),
                                     timeout=rclpy.duration.Duration(seconds=5.0)).transform
        q = t.rotation
        x, y, z, w = q.x, q.y, q.z, q.w
        rot = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        return np.array([t.translation.x, t.translation.y, t.translation.z]), rot

    def in_view(self, x, y, base_z, cup):
        """True if the whole cup (with handle reach) projects inside the image."""
        cam, rot = self.camera()
        k = self.info.k
        r = cup['outer_diameter'] / 2 + 0.02
        ring = [(x + r * math.cos(a), y + r * math.sin(a), base_z + h)
                for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)
                for h in (0.0, cup['height'])]
        p = (np.array(ring) - cam) @ rot  # world -> optical
        u = k[0] * p[:, 0] / p[:, 2] + k[2]
        v = k[4] * p[:, 1] / p[:, 2] + k[5]
        return bool(np.all(p[:, 2] > 0) and np.all((u >= 0) & (u < self.info.width)
                                                   & (v >= 0) & (v < self.info.height)))

    def collect(self, n, since_ns, wall_timeout):
        end = time.monotonic() + wall_timeout
        while time.monotonic() < end:
            with self.lock:
                got = [f for f in self.frames if f[0] >= since_ns]
            if len(got) >= n:
                return got[:n]
            time.sleep(0.1)
        return got

    def move_arm(self, group, poses):
        if self.moveit is None:
            # Uses cup_task's MoveIt client (only needed for cases with an arm).
            from cup_task.moveit_client import MoveItClient
            self.moveit = MoveItClient(self, 'world', 0.5, 0.5, 10.0)
            self.moveit.wait_ready()
        for name in poses:
            self.moveit.move_to_joints(group, self.moveit.named_pose(group, name))


# ---- scoring ----------------------------------------------------------------------
def score(frames, cups, match, wrong_deg):
    """Per-cup statistics over the frames; cups = list of truth dicts."""
    stats = [{'found': 0, 'xy': [], 'z': [], 'seen': 0, 'yaw': [], 'wrong': 0} for _ in cups]
    false_cups = 0
    for _, est in frames:
        pairs = sorted((math.hypot(e['x'] - c['x'], e['y'] - c['y']), ci, ei)
                       for ci, c in enumerate(cups) for ei, e in enumerate(est))
        used_c, used_e = set(), set()
        for d, ci, ei in pairs:
            if d > match or ci in used_c or ei in used_e:
                continue
            used_c.add(ci)
            used_e.add(ei)
            s, e, c = stats[ci], est[ei], cups[ci]
            s['found'] += 1
            s['xy'].append(d * 1000)
            s['z'].append((e['z'] - c['z']) * 1000)
            if e['visible']:
                s['seen'] += 1
                err = wrap_deg(math.degrees(e['yaw'] - c['yaw']))
                s['yaw'].append(abs(err))
                if abs(err) > wrong_deg or c['geo_visible'] == 0.0:
                    s['wrong'] += 1
        false_cups += len(est) - len(used_e)
    return stats, false_cups


def fmt(v, digits=1):
    return '' if v is None else f'{v:.{digits}f}'


def stat_row(s, nframes):
    def agg(values, fn):
        return float(fn(values)) if len(values) else None
    return {
        'frames': nframes,
        'detected_pct': 100.0 * s['found'] / nframes if nframes else None,
        'xy_err_mean_mm': agg(s['xy'], np.mean), 'xy_err_max_mm': agg(s['xy'], np.max),
        'z_err_mean_mm': agg(s['z'], np.mean),
        'z_err_max_abs_mm': agg(np.abs(s['z']), np.max),
        'handle_seen_pct': 100.0 * s['seen'] / s['found'] if s['found'] else None,
        'yaw_err_mean_deg': agg(s['yaw'], np.mean), 'yaw_err_max_deg': agg(s['yaw'], np.max),
        'wrong_visible': s['wrong'],
    }


# ---- the evaluation -----------------------------------------------------------------
class Run:

    def __init__(self, node, cfg, cup, out):
        self.node, self.cfg, self.cup, self.out = node, cfg, cup, out
        self.s = cfg['settings']
        # YAML reads a bare off/on as false/true: accept both spellings.
        self.s['noise_modes'] = ['on' if m in (True, 'on') else 'off'
                                 for m in self.s['noise_modes']]
        self.log_lines = []

    def log(self, text=''):
        print(text, flush=True)
        self.log_lines.append(text)

    def setup_cups(self, specs):
        """Place cups per spec (unused ones parked); return their truth dicts."""
        n = self.node
        names = [FIRST_CUP] + [f'eval_cup_{i}' for i in range(1, 3)]
        n.ensure_cups(names[:max(len(specs), 1)])
        cam, _ = n.camera()
        for i, name in enumerate(names):
            if i >= len(specs):
                if name in n.truth:
                    n.park(name, i, self.s['park_xy'])
                continue
            x, y = specs[i]['xy']
            bearing = math.atan2(y - cam[1], x - cam[0])
            yaw = (math.radians(specs[i]['yaw']) if 'yaw' in specs[i]
                   else bearing + math.radians(specs[i]['yaw_rel']))
            n.place(name, x, y, yaw)
        n.sim_sleep(self.s['settle_time'])
        return [self.truth_of(names[i], cam) for i in range(len(specs))]

    def truth_of(self, name, cam):
        x, y, z, yaw = self.node.truth[name]
        bearing = math.atan2(y - cam[1], x - cam[0])
        mid = np.array([x, y, z + self.cup['height'] / 2])
        return {'name': name, 'x': x, 'y': y, 'z': z, 'yaw': yaw,
                'yaw_rel': wrap_deg(math.degrees(yaw - bearing)),
                'distance': float(np.linalg.norm(mid - cam)),
                'geo_visible': handle_visible_fraction(self.cup, x, y, z, yaw, cam),
                'in_view': self.node.in_view(x, y, z, self.cup)}

    def frames(self, nframes):
        since = self.node.now_ns()
        with self.node.lock:
            self.node.robot_pts.clear()
        got = self.node.collect(nframes, since, 60 + 8 * nframes)
        with self.node.lock:
            pts = list(self.node.robot_pts)
        self.robot_median = int(np.median(pts)) if pts else 0
        if len(got) < nframes:
            self.log(f'    only {len(got)}/{nframes} estimator frames arrived')
        return got

    def run_cases(self, only):
        rows = []
        for case in self.cfg['cases']:
            if only and case['name'] not in only:
                continue
            self.log(f'== {case["name"]}')
            arm = case.get('arm')
            try:
                cups = self.setup_cups(case['cups'])
                if arm:
                    self.node.move_arm(arm['group'], arm['poses'])
                    self.node.sim_sleep(self.s['settle_time'])
                    cam, _ = self.node.camera()
                    cups = [self.truth_of(c['name'], cam) for c in cups]
                for mode in self.s['noise_modes']:
                    self.node.set_noise(mode == 'on')
                    self.node.sim_sleep(self.s['settle_time'])
                    got = self.frames(self.s['frames'])
                    stats, false_cups = score(got, cups, self.s['match_distance'],
                                              self.s['wrong_visible_deg'])
                    for i, (c, st) in enumerate(zip(cups, stats)):
                        row = {'case': case['name'], 'noise': mode, 'cup': i,
                               'x': c['x'], 'y': c['y'], 'distance_m': c['distance'],
                               'handle_yaw_deg': math.degrees(c['yaw']),
                               'handle_rel_deg': c['yaw_rel'],
                               'handle_geo_visible_pct': 100 * c['geo_visible'],
                               'in_view': c['in_view'], 'arm_over_table': bool(arm),
                               'robot_points_removed': self.robot_median,
                               **stat_row(st, len(got)), 'false_cups_in_case': false_cups}
                        rows.append(row)
                        self.log('    ' + self.row_text(row))
            except Exception as e:  # keep going with the other cases
                self.log(f'    FAILED: {e}')
            finally:
                if arm:
                    try:
                        self.node.move_arm(arm['group'], arm['back'])
                    except Exception as e:
                        self.log(f'    could not move the arm back: {e}')
        return rows

    @staticmethod
    def row_text(r):
        return (f'noise {r["noise"]:3s} cup {r["cup"]} d {r["distance_m"]:.2f} m  '
                f'detected {fmt(r["detected_pct"], 0)}%  xy {fmt(r["xy_err_mean_mm"])}/'
                f'{fmt(r["xy_err_max_mm"])} mm  z {fmt(r["z_err_mean_mm"])} mm  handle '
                f'(rel {r["handle_rel_deg"]:+.0f}, geo {r["handle_geo_visible_pct"]:.0f}%) '
                f'seen {fmt(r["handle_seen_pct"], 0)}%  yaw {fmt(r["yaw_err_mean_deg"])}/'
                f'{fmt(r["yaw_err_max_deg"])} deg  wrong {r["wrong_visible"]}  false '
                f'{r["false_cups_in_case"]}' + ('' if r['in_view'] else '  [partly out of view]')
                + (f'  robot points removed {r["robot_points_removed"]}'
                   if r['arm_over_table'] else ''))

    def run_sweep(self):
        sw = self.cfg.get('visibility_sweep')
        rows = []
        if not sw:
            return rows
        angles = np.arange(-180, 180, sw['step_deg'])
        for mode in self.s['noise_modes']:
            self.node.set_noise(mode == 'on')
            for pos in sw['positions']:
                self.log(f'== sweep at {pos}, noise {mode}')
                for rel in angles:
                    try:
                        cups = self.setup_cups([{'xy': pos, 'yaw_rel': float(rel)}])
                        got = self.frames(sw['frames'])
                        st, false_cups = score(got, cups, self.s['match_distance'],
                                               self.s['wrong_visible_deg'])
                    except Exception as e:
                        self.log(f'    {rel:+4.0f} deg FAILED: {e}')
                        continue
                    c = cups[0]
                    row = {'x': pos[0], 'y': pos[1], 'noise': mode,
                           'handle_rel_deg': c['yaw_rel'], 'distance_m': c['distance'],
                           'handle_geo_visible_pct': 100 * c['geo_visible'],
                           **stat_row(st[0], len(got)), 'false_cups': false_cups}
                    rows.append(row)
                    self.log(f'    rel {c["yaw_rel"]:+5.0f} deg  geo '
                             f'{row["handle_geo_visible_pct"]:3.0f}%'
                             f'  seen {fmt(row["handle_seen_pct"], 0):>3s}%  yaw err '
                             f'{fmt(row["yaw_err_mean_deg"])}  detected '
                             f'{fmt(row["detected_pct"], 0)}%')
        return rows


def hidden_summary(rows):
    """Text: the handle angles at which the handle was reported "not seen"."""
    out = []
    groups = {}
    for r in rows:
        groups.setdefault((r['x'], r['y'], r['noise']), []).append(r)
    for (x, y, mode), rs in groups.items():
        rs = [r for r in rs if r['handle_seen_pct'] is not None]
        hidden = sorted(r['handle_rel_deg'] for r in rs if r['handle_seen_pct'] < 50)
        seen = [abs(r['handle_rel_deg']) for r in rs if r['handle_seen_pct'] >= 50]
        dist = np.mean([r['distance_m'] for r in rs]) if rs else float('nan')
        out.append(f'  cup at ({x:.2f}, {y:.2f}), {dist:.2f} m from the camera, noise {mode}:')
        out.append('    reported hidden (< 50% of frames) at '
                   + (', '.join(f'{a:+.0f}' for a in hidden) or 'no angle') + ' deg')
        if hidden:
            out.append(f'    -> a hidden handle points at most '
                       f'{max(abs(a) for a in hidden):.0f} deg away from "straight away '
                       f'from the camera"')
        if seen:
            out.append(f'    -> reported seen at every angle with |angle| >= '
                       f'{min(seen):.0f} deg that was tested'
                       if not any(abs(a) >= min(seen) for a in hidden) else
                       f'    -> seen and hidden angles overlap (smallest seen |angle| '
                       f'{min(seen):.0f} deg)')
    return out


# ---- outputs --------------------------------------------------------------------------
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
MODE_STYLE = {'off': ('#2a78d6', 'o', 'noise off'), 'on': ('#eb6834', '^', 'noise on')}


def write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)


def plot_distance(path, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=SURFACE)
    _style(ax)
    for mode, (color, marker, label) in MODE_STYLE.items():
        rs = [r for r in rows if r['noise'] == mode and r['xy_err_mean_mm'] is not None]
        if not rs:
            continue
        d = np.array([r['distance_m'] for r in rs]) + (0.004 if mode == 'on' else -0.004)
        mean = np.array([r['xy_err_mean_mm'] for r in rs])
        top = np.array([r['xy_err_max_mm'] for r in rs])
        ax.vlines(d, mean, top, color=color, lw=2, alpha=0.6)
        ax.plot(d, mean, marker, color=color, ms=8, mec=SURFACE, mew=2, ls='none',
                label=f'{label}: mean (line up to max)')
    ax.set_xlabel('distance from the camera to the cup (m)', color=INK)
    ax.set_ylabel('position error (mm)', color=INK)
    ax.set_ylim(bottom=0)
    ax.set_title('Cup position error vs distance from the camera (one point per cup and case)',
                 color=INK, fontsize=11, loc='left')
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc='upper left')
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_angle(path, sweep_rows, case_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True, facecolor=SURFACE)
    for ax in (top, bottom):
        _style(ax)
    positions = sorted({(r['x'], r['y']) for r in sweep_rows})
    dashes = ['-', '--', ':']
    for pi, pos in enumerate(positions):
        rs = sorted((r for r in sweep_rows if (r['x'], r['y']) == pos and r['noise'] == 'off'),
                    key=lambda r: r['handle_rel_deg'])
        if rs:
            dist = np.mean([r['distance_m'] for r in rs])
            top.plot([r['handle_rel_deg'] for r in rs],
                     [r['handle_geo_visible_pct'] for r in rs], color=INK2, lw=1.5,
                     ls=dashes[pi % 3], label=f'share of handle surface in view, cup at '
                     f'{dist:.2f} m (geometry)')
        for mode, (color, marker, label) in MODE_STYLE.items():
            rs = sorted((r for r in sweep_rows if (r['x'], r['y']) == pos and r['noise'] == mode
                         and r['handle_seen_pct'] is not None),
                        key=lambda r: r['handle_rel_deg'])
            if not rs:
                continue
            dist = np.mean([r['distance_m'] for r in rs])
            a = [r['handle_rel_deg'] for r in rs]
            top.plot(a, [r['handle_seen_pct'] for r in rs], color=color, lw=2,
                     ls=dashes[pi % 3], marker=marker, ms=6 if pi else 8, mec=SURFACE,
                     label=f'reported "seen", {label}, cup at {dist:.2f} m')
            yr = [r for r in rs if r['yaw_err_mean_deg'] is not None]
            bottom.vlines([r['handle_rel_deg'] for r in yr], [r['yaw_err_mean_deg'] for r in yr],
                          [r['yaw_err_max_deg'] for r in yr], color=color, lw=2, alpha=0.5)
            bottom.plot([r['handle_rel_deg'] for r in yr], [r['yaw_err_mean_deg'] for r in yr],
                        marker, color=color, ms=6 if pi else 8, mec=SURFACE, ls='none',
                        label=f'{label}, cup at {dist:.2f} m')
    for mode, (color, marker, label) in MODE_STYLE.items():
        yr = [r for r in case_rows if r['noise'] == mode and r['yaw_err_mean_deg'] is not None]
        if yr:
            bottom.plot([r['handle_rel_deg'] for r in yr], [r['yaw_err_mean_deg'] for r in yr],
                        marker, mfc='none', mec=color, ms=9, mew=1.5, ls='none',
                        label=f'{label}, test cases (hollow)')
    top.set_ylabel('% of frames / % of surface', color=INK)
    top.set_ylim(-3, 103)
    top.set_title('Handle visibility and yaw error vs handle angle', color=INK, fontsize=11,
                  loc='left')
    top.legend(frameon=False, fontsize=8, labelcolor=INK, loc='upper left',
               bbox_to_anchor=(1.01, 1.0))
    bottom.set_ylabel('handle yaw error (deg)\nmean, line up to max', color=INK)
    bottom.set_ylim(bottom=0)
    bottom.set_xlabel('handle angle relative to the camera (deg): 0 = pointing straight away '
                      'from it, ±180 = straight at it, +90 = to the left', color=INK)
    bottom.set_xticks(range(-180, 181, 45))
    bottom.legend(frameon=False, fontsize=8, labelcolor=INK, loc='upper left',
                  bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def read_csv(path):
    """Rows of a results CSV with numbers, booleans and blanks converted back."""
    if not os.path.exists(path):
        return []

    def conv(v):
        if v == '':
            return None
        if v in ('True', 'False'):
            return v == 'True'
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            return v
    with open(path) as f:
        return [{k: conv(v) for k, v in row.items()} for row in csv.DictReader(f)]


def make_plots(out, case_rows, sweep_rows):
    if case_rows:
        plot_distance(os.path.join(out, 'error_vs_distance.png'), case_rows)
    if sweep_rows or case_rows:
        plot_angle(os.path.join(out, 'handle_vs_angle.png'), sweep_rows, case_rows)


def default_paths():
    """Workspace source files if they exist (editable without rebuilding)."""
    ws = os.path.dirname(os.path.dirname(get_package_prefix('cup_perception')))
    src = os.path.join(ws, 'src', 'cup_task_ws')
    cases = os.path.join(src, 'cup_perception', 'config', 'perception_eval.yaml')
    if not os.path.exists(cases):
        cases = os.path.join(get_package_share_directory('cup_perception'), 'config',
                             'perception_eval.yaml')
    out = os.path.join(src, 'results') if os.path.isdir(src) else os.path.abspath('results')
    return cases, out


def main():
    cases_default, out_default = default_paths()
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--cases', default=cases_default)
    ap.add_argument('--out', default=out_default)
    ap.add_argument('--only', nargs='*', help='run only these case names')
    ap.add_argument('--spawn-z', type=float, default=0.40,
                    help='robot mounting height, as passed to sim.launch.py')
    ap.add_argument('--no-sweep', action='store_true')
    ap.add_argument('--no-cases', action='store_true')
    ap.add_argument('--replot', action='store_true',
                    help='only redraw the plots from the CSV files in --out')
    args, ros_args = ap.parse_known_args()
    SPAWN_XYZ[2] = args.spawn_z
    if args.replot:
        case_rows = read_csv(os.path.join(args.out, 'cases.csv'))
        sweep_rows = read_csv(os.path.join(args.out, 'sweep.csv'))
        if not case_rows and not sweep_rows:
            raise SystemExit(f'no cases.csv or sweep.csv in {args.out}')
        make_plots(args.out, case_rows, sweep_rows)
        print(f'Plots redrawn in {args.out}')
        return
    with open(args.cases) as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(get_package_share_directory('cup_description'), 'config',
                           'cup.yaml')) as f:
        cup = yaml.safe_load(f)['cup']
    os.makedirs(args.out, exist_ok=True)

    rclpy.init(args=ros_args)
    node = Evaluator()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    run = Run(node, cfg, cup, args.out)
    t0 = time.monotonic()
    end = time.monotonic() + 60
    while (node.info is None or FIRST_CUP not in node.truth) and time.monotonic() < end:
        time.sleep(0.2)
    if node.info is None or FIRST_CUP not in node.truth:
        raise SystemExit('no camera_info or Gazebo poses: is the simulation running '
                         '(same ROS_DOMAIN_ID / IGN_PARTITION)?')
    run.log(f'Test set: {args.cases}')
    case_rows = [] if args.no_cases else run.run_cases(args.only)
    sweep_rows = [] if args.no_sweep else run.run_sweep()
    node.set_noise(False)
    # A partial rerun (--only, --no-sweep, --no-cases) replaces only what it
    # measured; the other rows of the earlier results are kept.
    names = [c['name'] for c in cfg['cases']]
    if args.only or args.no_cases:
        redone = {r['case'] for r in case_rows}
        old = [r for r in read_csv(os.path.join(args.out, 'cases.csv'))
               if r['case'] not in redone and r['case'] in names]
        case_rows = sorted(old + case_rows, key=lambda r: (names.index(r['case']),
                                                           r['noise'] != 'off', r['cup']))
    if args.no_sweep:
        sweep_rows = read_csv(os.path.join(args.out, 'sweep.csv'))
    write_csv(os.path.join(args.out, 'cases.csv'), case_rows)
    write_csv(os.path.join(args.out, 'sweep.csv'), sweep_rows)
    run.log('')
    run.log('Summary (position error mean/max mm, yaw error mean/max deg when "seen"):')
    run.log(f'{"case":17s} {"noise":5s} {"cup":3s} {"dist m":>6s} {"det %":>5s} '
            f'{"xy mean":>7s} {"xy max":>6s} {"z mean":>6s} {"rel deg":>7s} {"geo %":>5s} '
            f'{"seen %":>6s} {"yaw mean":>8s} {"yaw max":>7s} {"wrong":>5s} {"false":>5s}')
    for r in case_rows:
        run.log(f'{r["case"]:17s} {r["noise"]:5s} {r["cup"]:3d} {r["distance_m"]:6.2f} '
                f'{fmt(r["detected_pct"], 0):>5s} {fmt(r["xy_err_mean_mm"]):>7s} '
                f'{fmt(r["xy_err_max_mm"]):>6s} {fmt(r["z_err_mean_mm"]):>6s} '
                f'{r["handle_rel_deg"]:+7.0f} {r["handle_geo_visible_pct"]:5.0f} '
                f'{fmt(r["handle_seen_pct"], 0):>6s} {fmt(r["yaw_err_mean_deg"]):>8s} '
                f'{fmt(r["yaw_err_max_deg"]):>7s} {r["wrong_visible"]:5d} '
                f'{r["false_cups_in_case"]:5d}' + ('' if r['in_view'] else '  partly out of view'))
    if sweep_rows:
        run.log('')
        run.log('Handle visibility sweep:')
        for line in hidden_summary(sweep_rows):
            run.log(line)
    make_plots(args.out, case_rows, sweep_rows)
    run.log(f'\nResults in {args.out} ({(time.monotonic() - t0) / 60:.0f} min)')
    with open(os.path.join(args.out, 'summary.txt'), 'w') as f:
        f.write('\n'.join(run.log_lines) + '\n')
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
