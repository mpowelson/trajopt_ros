cmake_minimum_required(VERSION 2.8.3)
project(trajopt_examples)

add_compile_options(-std=c++11 -Wall -Wextra)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  roslib
  trajopt
  tesseract_ros
  tesseract_planning
  octomap_ros
  pcl_conversions
)

find_package(pagmo REQUIRED)

find_package(PCL REQUIRED COMPONENTS core features filters io segmentation surface)

catkin_package()

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  SYSTEM ${PCL_INCLUDE_DIRS}
)

add_executable(pagmo_example src/pagmo_example.cpp)
target_link_libraries(pagmo_example ${catkin_LIBRARIES} Pagmo::pagmo)
target_compile_options(pagmo_example PRIVATE -Wsuggest-override -Wconversion -Wsign-conversion)

add_executable(pagmo_example2 src/pagmo_example2.cpp)
target_link_libraries(pagmo_example2 ${catkin_LIBRARIES})
target_compile_options(pagmo_example2 PRIVATE -Wsuggest-override -Wconversion -Wsign-conversion)

add_executable(pagmo_ipopt_example src/pagmo_ipopt_example.cpp)
target_link_libraries(pagmo_ipopt_example ${catkin_LIBRARIES} Pagmo::pagmo)
target_compile_options(pagmo_ipopt_example PRIVATE -Wsuggest-override -Wconversion -Wsign-conversion)

add_executable(${PROJECT_NAME}_basic_cartesian_plan src/basic_cartesian_plan.cpp)
target_link_libraries(${PROJECT_NAME}_basic_cartesian_plan ${PCL_LIBRARIES} ${catkin_LIBRARIES})
target_compile_options(${PROJECT_NAME}_basic_cartesian_plan PRIVATE -Wsuggest-override -Wconversion -Wsign-conversion)

add_executable(${PROJECT_NAME}_glass_up_right_plan src/glass_up_right_plan.cpp)
target_link_libraries(${PROJECT_NAME}_glass_up_right_plan ${catkin_LIBRARIES} Pagmo::pagmo)
target_compile_options(${PROJECT_NAME}_glass_up_right_plan PRIVATE -Wsuggest-override -Wconversion -Wsign-conversion)

add_executable(${PROJECT_NAME}_glass_up_right_plan_ipopt src/glass_up_right_plan_ipopt.cpp)
target_link_libraries(${PROJECT_NAME}_glass_up_right_plan_ipopt ${catkin_LIBRARIES} Pagmo::pagmo)
target_compile_options(${PROJECT_NAME}_glass_up_right_plan_ipopt PRIVATE -Wsuggest-override -Wconversion -Wsign-conversion)

add_executable(${PROJECT_NAME}_glass_up_right_plan_nlopt src/glass_up_right_plan_nlopt.cpp)
target_link_libraries(${PROJECT_NAME}_glass_up_right_plan_nlopt ${catkin_LIBRARIES} Pagmo::pagmo)
target_compile_options(${PROJECT_NAME}_glass_up_right_plan_nlopt PRIVATE -Wsuggest-override -Wconversion -Wsign-conversion)

add_executable(${PROJECT_NAME}_puzzle_piece_plan src/puzzle_piece_plan.cpp)
target_link_libraries(${PROJECT_NAME}_puzzle_piece_plan ${catkin_LIBRARIES})
target_compile_options(${PROJECT_NAME}_puzzle_piece_plan PRIVATE -Wsuggest-override -Wconversion -Wsign-conversion)

add_executable(${PROJECT_NAME}_car_seat_demo src/car_seat_demo.cpp)
target_link_libraries(${PROJECT_NAME}_car_seat_demo ${catkin_LIBRARIES})
target_compile_options(${PROJECT_NAME}_car_seat_demo PRIVATE -Wsuggest-override -Wconversion -Wsign-conversion)

add_executable(${PROJECT_NAME}_puzzle_piece_auxillary_axes_plan src/puzzle_piece_auxillary_axes_plan.cpp)
target_link_libraries(${PROJECT_NAME}_puzzle_piece_auxillary_axes_plan ${catkin_LIBRARIES})
target_compile_options(${PROJECT_NAME}_puzzle_piece_auxillary_axes_plan PRIVATE -Wsuggest-override -Wconversion -Wsign-conversion)

add_executable(${PROJECT_NAME}_pick_and_place_plan src/pick_and_place_plan.cpp)
target_link_libraries(${PROJECT_NAME}_pick_and_place_plan ${catkin_LIBRARIES})

# Mark executables and/or libraries for installation
install(
  TARGETS ${PROJECT_NAME}_basic_cartesian_plan ${PROJECT_NAME}_glass_up_right_plan ${PROJECT_NAME}_puzzle_piece_plan ${PROJECT_NAME}_car_seat_demo ${PROJECT_NAME}_puzzle_piece_auxillary_axes_plan ${PROJECT_NAME}_pick_and_place_plan
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY config launch meshes urdf
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

