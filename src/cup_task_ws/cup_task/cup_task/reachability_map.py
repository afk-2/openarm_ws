"""
Reachability map: where on the table can pick_place pick up the cup?

Offline (MoveIt only, no Gazebo physics). For a grid of cup positions over
the table and several handle directions, checks whether pick_place's whole
grasp sequence is feasible, with pick_place's own code and settings:
grasp IK (grasp_search_group), the follow-on poses with the arm group, every
straight-line move, the open fingers clear of the handle at the grasp and
during the descent, and the closed fingers at least handle_min_clearance
from it. The handle is taken as seen (its direction known).

Run with reachability.launch.py, which starts move_group and
robot_state_publisher for the given mounting height and then this script:

    ros2 launch cup_task reachability.launch.py spawn_z:=0.40

Writes <out>/reachability_<tag>.csv and .png (success rate per position as a
heatmap over the table, with the camera's field of view). With --plot-only
it redraws one figure comparing all CSVs given with --compare.

Speed: the handle-independent part of each grasp (reach, follow-on poses,
straight lines, checked with the cup body only) is computed once per
position and reused for every handle direction.
"""
import argparse
import csv
import math
import os
import threading
import time

import numpy as np
import rclpy
import tf2_ros
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import MultiThreadedExecutor

from cup_task import grasps
from cup_task.moveit_client import TaskError, pose_msg
from cup_task.pick_place import PickPlace, UP

TABLE_TOP = 0.75                       # above the floor (cup_table.sdf)
TABLE_X, TABLE_Y = (0.12, 0.92), (-0.6, 0.6)
CAMERA_FRAME = 'camera_color_optical_frame'
IMAGE_W, IMAGE_H, HFOV = 640, 480, 1.2043   # cup_description/urdf/d435.xacro
# Quick geometric bound, to skip positions no grasp can reach: from the right
# shoulder (joint 2, robot frame) the URDF's links add up to 0.44 m to the
# wrist plus 0.164 m to the finger pads (gripper config); 0.66 m adds margin.
SHOULDER = np.array([0.0, -0.1525, 0.697])
MAX_REACH = 0.66


class Mapper:

    def __init__(self, node, table_z):
        self.n = node
        self.table_z = table_z
        g = node.gripper
        self.g = g
        lifted = node.moveit.named_pose(g.arm_group, node.lifted_pose)
        node.arm_joints = list(lifted)
        self.seed = node.arm_seed({**node.joints, **lifted}, g.max_opening)

    # ---- per position: handle-independent feasibility ----------------------
    def sequence(self, grasp, base):
        """Full sequence for one grasp with the cup body only, or None."""
        n, g = self.n, self.g
        quat = grasps.quat_from_matrix(grasp.rotation)
        ee = grasps.ee_position(grasp, base, g.geometry)
        sol = n.moveit.solve_ik(n.search_group or g.arm_group, g.ee_link, ee, quat,
                                self.seed, attempts=2, joints=n.arm_joints,
                                limit_margin=n.limit_margin)
        if sol is None:
            return None, 'grasp pose unreachable'
        place_base = np.array([n.place_xy[0], n.place_xy[1], base[2]])
        held = n.cup.outer_diameter
        back = -grasp.approach * n.approach_distance
        poses = {
            'pre-grasp': (ee + back + UP * n.approach_height, g.max_opening),
            'grasp': (ee, g.max_opening),
            'lift': (ee + UP * n.lift_height, held),
            'above place': (ee - base + place_base + UP * n.lift_height, held),
            'place': (ee - base + place_base + UP * n.place_drop, held),
            'retreat': (ee - base + place_base - grasp.approach * n.retreat_distance
                        + UP * n.place_drop, g.max_opening),
        }
        solutions, seed = {'grasp': sol}, sol
        for name in ('pre-grasp', 'lift', 'above place', 'place', 'retreat'):
            pos, opening = poses[name]
            s = n.moveit.solve_ik(g.arm_group, g.ee_link, pos, quat, n.arm_seed(seed, opening),
                                  attempts=4, joints=n.arm_joints, limit_margin=n.limit_margin)
            if s is None:
                return None, f'{name} pose unreachable'
            solutions[name], seed = s, s
        blocked = n.blocked_straight_move(poses, quat, solutions)
        if blocked:
            return None, f'straight-line {blocked.split(":")[0]}'
        return (quat, poses, solutions), 'ok'

    # ---- per handle direction ----------------------------------------------
    def handle_ok(self, grasp, base, yaw, seq):
        """Handle-specific checks for a grasp whose sequence is feasible."""
        n, g = self.n, self.g
        quat, poses, solutions = seq
        n.moveit.add_object(n.cup_object(base, yaw))
        # Open fingers at the grasp, and the descent, with the handle present.
        open_state = n.arm_seed(solutions['grasp'], g.max_opening)
        if any(p for p in n.moveit.contacts(g.arm_group, open_state)):
            return False, 'open fingers hit the handle'
        frac, _ = n.moveit.straight_path(
            g.arm_group, g.ee_link, n.arm_seed(solutions['pre-grasp'], g.max_opening),
            [pose_msg(poses['grasp'][0], quat)], True)
        if frac < n.min_fraction:
            return False, 'descent hits the handle'
        gaps = n.handle_clearance(base, yaw, solutions['grasp'], n.handle_offsets(False))
        n.moveit.add_object(n.cup_object(base, yaw))
        if min(gaps.values()) < n.min_clearance:
            return False, 'closed fingers too close to the handle'
        return True, 'ok'

    def position(self, x, y, yaws):
        """Per handle direction: (feasible, reason), plus timing."""
        n = self.n
        base = np.array([x, y, self.table_z])
        body = n.cup_object(base, 0.0)
        del body.primitives[1:], body.primitive_poses[1:]   # body only
        n.moveit.add_object(body)
        headings = np.arange(0.0, 360.0, n.heading_step)
        cands = grasps.candidates(n.cup, self.g.geometry, n.height_fraction,
                                  n.table_clearance, n.pitches, headings)
        cache = {}
        out = []
        for yaw in yaws:
            offsets = n.handle_offsets(False)
            usable = [c for c in cands
                      if min(grasps.angle_to_closing_line(yaw + math.radians(o), c)
                             for o in offsets) >= n.finger_min_angle]
            # Grasps already known to work first, then untested ones.
            usable.sort(key=lambda c: 0 if cache.get(c.name, (None,))[0] else 1)
            result, why = False, 'no grasp keeps the fingers away from the handle'
            reasons = {}
            for c in usable:
                if c.name not in cache:
                    n.moveit.add_object(body)
                    cache[c.name] = self.sequence(c, base)
                seq, reason = cache[c.name]
                if seq is None:
                    reasons[reason] = reasons.get(reason, 0) + 1
                    continue
                ok, reason = self.handle_ok(c, base, yaw, seq)
                n.moveit.add_object(body)
                if ok:
                    result, why = True, c.name
                    break
                reasons[reason] = reasons.get(reason, 0) + 1
            if not result and reasons:
                why = max(reasons, key=reasons.get)
            out.append((result, why))
        return out


def camera_footprint(tf_buffer, table_z):
    """Camera position and the image border projected onto the table plane."""
    t = tf_buffer.lookup_transform('world', CAMERA_FRAME, rclpy.time.Time(),
                                   timeout=rclpy.duration.Duration(seconds=10.0)).transform
    q = t.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    rot = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    cam = np.array([t.translation.x, t.translation.y, t.translation.z])
    f = IMAGE_W / 2 / math.tan(HFOV / 2)
    border = [(u, 0) for u in np.linspace(0, IMAGE_W, 9)] + \
        [(IMAGE_W, v) for v in np.linspace(0, IMAGE_H, 7)] + \
        [(u, IMAGE_H) for u in np.linspace(IMAGE_W, 0, 9)] + \
        [(0, v) for v in np.linspace(IMAGE_H, 0, 7)]
    pts = []
    for u, v in border:
        ray = rot @ np.array([(u - IMAGE_W / 2) / f, (v - IMAGE_H / 2) / f, 1.0])
        if ray[2] >= -1e-6:
            continue  # above the horizon
        s = (table_z - cam[2]) / ray[2]
        pts.append(cam[:2] + s * ray[:2])
    return cam, rot, f, np.array(pts)


def cup_in_view(cam, rot, f, x, y, table_z, cup):
    """Whole cup (with handle reach) inside the image."""
    r = cup.radius + 0.02
    ring = np.array([(x + r * math.cos(a), y + r * math.sin(a), table_z + h)
                     for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)
                     for h in (0.0, cup.height)])
    p = (ring - cam) @ rot
    if np.any(p[:, 2] <= 0):
        return False
    u = f * p[:, 0] / p[:, 2] + IMAGE_W / 2
    v = f * p[:, 1] / p[:, 2] + IMAGE_H / 2
    return bool(np.all((u >= 0) & (u < IMAGE_W) & (v >= 0) & (v < IMAGE_H)))


# ---- plotting ------------------------------------------------------------------
INK, INK2, SURFACE, GRID = '#0b0b0b', '#52514e', '#fcfcfb', '#e4e3df'


def read_rows(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ('x', 'y', 'success_rate', 'spawn_z'):
            r[k] = float(r[k])
        r['in_view'] = r['in_view'] == 'True'
        r['failure_reasons'] = r.get('failure_reasons') or ''
    return rows


def summary(rows, step, reliable=0.75):
    """Areas (m^2) where the success rate is >= reliable, in view or not."""
    cell = step * step
    good = [r for r in rows if r['success_rate'] >= reliable]
    return (len(good) * cell, sum(r['in_view'] for r in good) * cell,
            sum(r['success_rate'] > 0 for r in rows) * cell)


def plot(paths, out_png, step, place_xy):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    cmap = LinearSegmentedColormap.from_list(
        'blues', ['#f1f6fd', '#b9d3f3', '#6aa1e4', '#2a78d6', '#164f99', '#0b2d5c'])
    data = [(read_rows(p), p) for p in paths]
    fig, axes = plt.subplots(1, len(data), figsize=(5.0 * len(data) + 1.6, 7.0),
                             squeeze=False, facecolor=SURFACE)
    for ax, (rows, path) in zip(axes[0], data):
        z = rows[0]['spawn_z']
        ax.set_facecolor(SURFACE)
        ax.add_patch(Rectangle((TABLE_Y[0], TABLE_X[0]), TABLE_Y[1] - TABLE_Y[0],
                               TABLE_X[1] - TABLE_X[0], fill=False, ec=INK2, lw=1.0))
        for r in rows:
            skipped = 'geometric bound' in r['failure_reasons']
            ax.add_patch(Rectangle((r['y'] - step / 2, r['x'] - step / 2), step, step,
                                   fc=GRID if skipped else cmap(r['success_rate']),
                                   ec=SURFACE, lw=1.0))
        fov = np.array(yaml.safe_load(rows[0]['fov']))
        if len(fov):
            closed = np.vstack([fov, fov[:1]])
            ax.plot(closed[:, 1], closed[:, 0], color=INK, lw=1.6, ls='--',
                    label='camera field of view (table plane)')
        ax.plot([r['y'] for r in rows if not r['in_view']],
                [r['x'] for r in rows if not r['in_view']], 'x', color=INK2, ms=4, mew=0.8,
                label='cup not fully in view', ls='none')
        ax.plot(place_xy[1], place_xy[0], marker='*', color='#eb6834', ms=14, mec=INK,
                mew=0.8, ls='none', label='place spot')
        ax.add_patch(Rectangle((-0.15, -0.18), 0.30, 0.30, fc=GRID, ec=INK2, lw=0.8))
        ax.text(0.0, -0.03, 'robot', ha='center', va='center', fontsize=8, color=INK2)
        area, area_view, area_any = summary(rows, step)
        ax.set_title(f'mount {z:.2f} m (base {z - 0.40:+.2f} m)\n'
                     f'reliable area {area:.3f} m², {area_view:.3f} m² in view',
                     fontsize=10, color=INK, loc='left')
        ax.set_xlim(TABLE_Y[1] + 0.02, TABLE_Y[0] - 0.02)   # robot's left on the left
        ax.set_ylim(-0.22, TABLE_X[1] + 0.02)
        ax.set_aspect('equal')
        ax.set_xlabel('y (m), robot left ←', color=INK2)
        ax.set_ylabel('x (m), forward ↑', color=INK2)
        ax.tick_params(colors=INK2)
        for s in ax.spines.values():
            s.set_color(INK2)
    fig.subplots_adjust(left=0.07, right=0.86, bottom=0.17, top=0.82, wspace=0.3)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, cax=fig.add_axes([0.89, 0.25, 0.015, 0.5]))
    cb.set_label('share of handle directions with a feasible pick-and-place', color=INK2)
    cb.ax.tick_params(colors=INK2)
    handles, labels = axes[0][0].get_legend_handles_labels()
    handles.append(Rectangle((0, 0), 1, 1, fc=GRID, ec=INK2, lw=0.5))
    labels.append('beyond the arm\'s reach (not checked)')
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False, fontsize=9)
    fig.suptitle('Where can the right arm pick up the cup?\noffline MoveIt check of the full '
                 'pick-and-place; "reliable" = feasible for at least 75% of handle directions',
                 color=INK, fontsize=11)
    fig.savefig(out_png, dpi=130, facecolor=SURFACE)


# ---- main ------------------------------------------------------------------
def default_out():
    from ament_index_python.packages import get_package_prefix
    ws = os.path.dirname(os.path.dirname(get_package_prefix('cup_task')))
    src = os.path.join(ws, 'src', 'cup_task_ws')
    return os.path.join(src, 'results') if os.path.isdir(src) else os.path.abspath('results')


def main():
    ap = argparse.ArgumentParser(description='Reachability map for pick_place.')
    ap.add_argument('--spawn-z', type=float, default=0.40, help='robot mounting height (m)')
    ap.add_argument('--step', type=float, default=0.05, help='grid spacing (m)')
    ap.add_argument('--angles', type=int, default=8, help='handle directions per position')
    ap.add_argument('--x', type=float, nargs=2, default=[0.15, 0.80], help='x range (m)')
    ap.add_argument('--y', type=float, nargs=2, default=[-0.55, 0.40], help='y range (m)')
    ap.add_argument('--out', default=None)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--resume', action='store_true',
                    help='keep the positions already in the CSV and add the rest')
    ap.add_argument('--plot-only', action='store_true')
    ap.add_argument('--compare', nargs='*', default=[], help='CSV files for --plot-only')
    args, ros_args = ap.parse_known_args()
    out = args.out or default_out()
    os.makedirs(out, exist_ok=True)
    tag = args.tag or f'mount_{args.spawn_z:.2f}'
    share = os.path.join(get_package_share_directory('cup_task'), 'config')
    with open(os.path.join(share, 'pick_place.yaml')) as f:
        place_xy = yaml.safe_load(f)['pick_place']['ros__parameters']['place_xy']
    if args.plot_only:
        plot(args.compare, os.path.join(out, f'reachability_{tag}.png'), args.step, place_xy)
        return

    rclpy.init(args=['--ros-args', '--params-file', os.path.join(share, 'pick_place.yaml'),
                     '--params-file', os.path.join(share, 'gripper_openarm_pinch.yaml')]
               + [a for a in ros_args if a != '--ros-args'])
    node = PickPlace()
    node.get_logger().set_level(rclpy.logging.LoggingSeverity.WARN)
    tf_buffer = tf2_ros.Buffer()
    tf2_ros.TransformListener(tf_buffer, node)
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    node.moveit.wait_ready()
    time.sleep(2.0)
    table_z = TABLE_TOP - args.spawn_z
    mapper = Mapper(node, table_z)
    cam, rot, f, fov = camera_footprint(tf_buffer, table_z)
    print(f'mount {args.spawn_z:.2f} m: table at z {table_z:.3f} in the world frame, camera '
          f'{cam[2] - table_z:.2f} m above the table', flush=True)

    yaws = [2 * math.pi * k / args.angles for k in range(args.angles)]
    xs = np.arange(args.x[0], args.x[1] + 1e-9, args.step)
    ys = np.arange(args.y[0], args.y[1] + 1e-9, args.step)
    csv_path = os.path.join(out, f'reachability_{tag}.csv')
    fields = ['spawn_z', 'x', 'y', 'success_rate', 'in_view', 'feasible_angles_deg',
              'failure_reasons', 'seconds', 'fov']
    t_all = time.monotonic()
    done = set()
    if args.resume and os.path.exists(csv_path):
        done = {(round(r['x'], 3), round(r['y'], 3)) for r in read_rows(csv_path)}
        print(f'resuming: {len(done)} positions already in {csv_path}', flush=True)
    with open(csv_path, 'a' if done else 'w', newline='') as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=fields)
        if not done:
            w.writeheader()
        for x in xs:
            for y in ys:
                if not (TABLE_X[0] + 0.04 <= x <= TABLE_X[1] - 0.04
                        and TABLE_Y[0] + 0.04 <= y <= TABLE_Y[1] - 0.04):
                    continue
                if math.hypot(x - node.place_xy[0], y - node.place_xy[1]) < 0.08:
                    continue  # the place spot itself
                if (round(x, 3), round(y, 3)) in done:
                    continue
                t0 = time.monotonic()
                contact = np.array([x, y, table_z + node.height_fraction * node.cup.height])
                try:
                    if np.linalg.norm(contact - SHOULDER) > MAX_REACH:
                        res = [(False, 'beyond the arm\'s reach (geometric bound)')] * len(yaws)
                    else:
                        res = mapper.position(x, y, yaws)
                except TaskError as e:
                    res = [(False, f'error: {e}')] * len(yaws)
                rate = sum(ok for ok, _ in res) / len(res)
                reasons = {}
                for ok, why in res:
                    if not ok:
                        reasons[why] = reasons.get(why, 0) + 1
                w.writerow(dict(
                    spawn_z=args.spawn_z, x=round(x, 3), y=round(y, 3),
                    success_rate=round(rate, 3),
                    in_view=cup_in_view(cam, rot, f, x, y, table_z, node.cup),
                    feasible_angles_deg=' '.join(f'{math.degrees(a):.0f}' for a, (ok, _)
                                                 in zip(yaws, res) if ok),
                    failure_reasons='; '.join(f'{k} x{v}' for k, v in reasons.items()),
                    seconds=round(time.monotonic() - t0, 1),
                    fov=yaml.safe_dump(np.round(fov, 3).tolist(), default_flow_style=True)
                    .strip()))
                fcsv.flush()
                print(f'  ({x:.2f}, {y:+.2f}): {rate * 100:3.0f}% '
                      f'[{time.monotonic() - t0:.0f} s]', flush=True)
    print(f'done in {(time.monotonic() - t_all) / 60:.0f} min -> {csv_path}', flush=True)
    plot([csv_path], os.path.join(out, f'reachability_{tag}.png'), args.step, place_xy)
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
