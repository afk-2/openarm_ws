Start everything (Gazebo + robot + controllers + camera + RViz):
ros2 launch cup_bringup sim.launch.py

ros2 launch cup_bringup sim.launch.py rviz:=false              # Gazebo only
ros2 launch cup_bringup sim.launch.py gui:=false               # RViz only, no Gazebo window (faster)
ros2 launch cup_bringup sim.launch.py gui:=false rviz:=false   # fully headless

ros2 launch cup_bringup sim.launch.py camera_mast_height:=0.12 camera_tilt:=1.15 # launch and define mast dimensions.

After changing any code:
cd ~/openarm_ws
colcon build --symlink-install --packages-select cup_description cup_bringup cup_perception cup_task
source install/setup.bash

Useful checks (in a second terminal, after source install/setup.bash):
ros2 control list_controllers      # should show 5 active
ros2 topic list                    # all data streams
ros2 topic hz /camera/camera/depth/color/points   # is the camera publishing?