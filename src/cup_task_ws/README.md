# cup_task_ws

OpenArm v2.0 picks up a cup and puts it down level, in Gazebo Fortress (ROS 2 Humble).
How the packages fit together is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Run the simulation

Three terminals, each after `source ~/openarm_ws/install/setup.bash`:

```bash
# 1. Gazebo + robot + MoveIt (gui:=false runs without the Gazebo window)
ros2 launch cup_bringup sim.launch.py moveit:=true rviz:=false gui:=false

# 2. Cup pose from the camera, with D435-style depth noise
ros2 launch cup_perception cup_pose_estimator.launch.py depth_noise:=true cup_pose_topic:=/cup_pose

# 3. The task
ros2 launch cup_task pick_place.launch.py cup_pose_source:=none pose_from_perception:=true
```

Without the camera (Gazebo's true cup pose instead), skip terminal 2 and run
`ros2 launch cup_task pick_place.launch.py`.

## Simulation-only settings

Everything below exists to make Gazebo faster or more reliable. None of it
changes what the real robot does.

### Task (`cup_task`)

`pick_place.launch.py sim:=true` (the default) loads
[cup_task/config/pick_place_sim.yaml](cup_task/config/pick_place_sim.yaml) on
top of `pick_place.yaml`. On the real robot use `sim:=false`.

| Setting | Simulation | Real robot (`sim:=false`) |
|---|---|---|
| `use_sim_attach` | `true`: the cup is locked to the hand in Gazebo while held | `false`: the finger pads hold the cup |
| `gripper_close_mode` | `width`: close to the cup diameter minus `sim_grasp_width_under_diameter` (2 mm), then lock | `stall`: close until the fingers stall on the cup, then hold `grasp_squeeze` inside that width |
| `sim_missed_cup_distance` | missed cup = Gazebo's true cup position is more than 15 mm from the middle of the fingers | missed cup = the fingers stall narrower than 60% of the cup diameter |
| `sim_release_clearance` | release in three steps: open to cup diameter + 4 mm, unlock, open fully | the gripper just opens |
| `grasp_max_candidates` | `20`: check the full sequence for the 20 most comfortable grasps only | `0`: all of them |

### Gazebo and MoveIt (`cup_bringup/launch/sim.launch.py` arguments)

| Argument | Simulation default | Real robot |
|---|---|---|
| `camera_width`, `camera_height`, `camera_rate` | 640 x 480 at 2 Hz | set in the RealSense driver |
| `physics_step` | 0.002 s | no physics step |
| `grasp_ik_threads` | 1 thread for the grasp-search IK (pick_ik) | `kinematics.yaml` value (4); `move_group.launch.py` only changes it when this argument is given |
| `sim_cup_attach` | `true`: adds the Gazebo plugin that can lock the cup to the hand. The plugin locks the cup as soon as the robot loads; `cup_sim_attach_state.py` sends unlock until Gazebo confirms it, and keeps the lock state on `/cup_sim_attach/attached` so `cup_task` confirms every lock and unlock too | not in the robot description |
| `software_gl` | software rendering on WSL | not used |

### Robot description (only with `use_gazebo:=true`)

| Change | Simulation | Real robot |
|---|---|---|
| Finger end stop | closed stop moved 0.02 rad past 0 (`robot_description.py`): a Gazebo joint resting exactly on its limit stays stuck there | OpenArm's own limits |
| Collision meshes | left-arm meshes swapped for pre-mirrored copies (Gazebo cannot load negatively scaled meshes) | also used, same shapes |
| Camera | Gazebo RGB-D sensor in place of the RealSense driver | RealSense driver |

### Perception (`cup_perception`)

| Setting | Simulation | Real robot |
|---|---|---|
| `depth_noise:=true` (estimator launch) | adds D435-style errors to Gazebo's perfect depth | not used (real depth) |
| `gazebo_cup_pose` node | Gazebo's true cup pose on `/cup_pose` | not used (camera estimator) |
