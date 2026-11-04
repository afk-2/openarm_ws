cmake_minimum_required(VERSION 3.8)
project(cup_moveit_config)

find_package(ament_cmake REQUIRED)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS scripts/add_world_obstacles.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
