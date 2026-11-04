cmake_minimum_required(VERSION 3.8)
project(cup_description)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(openarm_description REQUIRED)

install(DIRECTORY urdf worlds config scripts
  DESTINATION share/${PROJECT_NAME}
)

# The cup's Gazebo model is generated from config/cup.yaml (dimensions are
# parameters, so the sim cup can be matched to the real one).
set(cup_model_dir "${CMAKE_CURRENT_BINARY_DIR}/models/ceramic_cup")
add_custom_command(
  OUTPUT "${cup_model_dir}/model.sdf"
  COMMAND ${Python3_EXECUTABLE} -I "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_cup_model.py"
          "${CMAKE_CURRENT_SOURCE_DIR}/config/cup.yaml" "${cup_model_dir}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_cup_model.py"
          "${CMAKE_CURRENT_SOURCE_DIR}/config/cup.yaml"
)
add_custom_target(cup_model ALL DEPENDS "${cup_model_dir}/model.sdf")
install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/models/"
  DESTINATION share/${PROJECT_NAME}/models
)

# cup_description.robot_description: builds the URDF for launch files.
ament_python_install_package(${PROJECT_NAME})

# Gazebo/DART rejects OpenArm's negative mesh scales (mirrored left arm), so
# ship pre-mirrored collision meshes; cup_description/robot_description.py
# swaps them in.
get_filename_component(openarm_description_SHARE "${openarm_description_DIR}/.." ABSOLUTE)
set(mirrored_dir "${CMAKE_CURRENT_BINARY_DIR}/openarm_mirrored")
add_custom_command(
  OUTPUT "${mirrored_dir}/.stamp"
  COMMAND ${Python3_EXECUTABLE} -I "${CMAKE_CURRENT_SOURCE_DIR}/scripts/mirror_meshes.py"
          "${openarm_description_SHARE}" "${mirrored_dir}"
  COMMAND ${CMAKE_COMMAND} -E touch "${mirrored_dir}/.stamp"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/scripts/mirror_meshes.py"
)
add_custom_target(mirror_meshes ALL DEPENDS "${mirrored_dir}/.stamp")
install(DIRECTORY "${mirrored_dir}/"
  DESTINATION share/${PROJECT_NAME}/meshes/openarm_mirrored
  PATTERN ".stamp" EXCLUDE
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
