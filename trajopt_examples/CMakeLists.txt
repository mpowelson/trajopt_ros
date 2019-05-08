cmake_minimum_required(VERSION 2.8.3)
project(trajopt_examples)

add_compile_options(-std=c++11 -Wall -Wextra)

find_package(rostest REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  roslib
  trajopt
  tesseract_ros
  tesseract_planning
  octomap_ros
  pcl_conversions
  rostest
)

find_package(PCL REQUIRED COMPONENTS core features filters io segmentation surface)

catkin_package()

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  SYSTEM ${PCL_INCLUDE_DIRS}
)

add_rostest_gtest(${PROJECT_NAME}_basic_cartesian_plan test/basic_cartesian_plan.test src/basic_cartesian_plan.cpp)
target_link_libraries(${PROJECT_NAME}_basic_cartesian_plan ${PCL_LIBRARIES} ${catkin_LIBRARIES})
target_compile_options(${PROJECT_NAME}_basic_cartesian_plan PRIVATE -Wsuggest-override -Wconversion -Wsign-conversion)

add_rostest_gtest(${PROJECT_NAME}_glass_up_right_plan test/glass_up_right_plan.test src/glass_up_right_plan.cpp)
target_link_libraries(${PROJECT_NAME}_glass_up_right_plan ${catkin_LIBRARIES})
target_compile_options(${PROJECT_NAME}_glass_up_right_plan PRIVATE -Wsuggest-override -Wconversion -Wsign-conversion)

add_rostest_gtest(${PROJECT_NAME}_puzzle_piece_plan test/puzzle_piece_plan.test src/puzzle_piece_plan.cpp)
target_link_libraries(${PROJECT_NAME}_puzzle_piece_plan ${catkin_LIBRARIES})
target_compile_options(${PROJECT_NAME}_puzzle_piece_plan PRIVATE -Wsuggest-override -Wconversion -Wsign-conversion)

add_rostest_gtest(${PROJECT_NAME}_car_seat_demo test/car_seat_demo.test src/car_seat_demo.cpp)
target_link_libraries(${PROJECT_NAME}_car_seat_demo ${catkin_LIBRARIES})
target_compile_options(${PROJECT_NAME}_car_seat_demo PRIVATE -Wsuggest-override -Wconversion -Wsign-conversion)

add_rostest_gtest(${PROJECT_NAME}_puzzle_piece_auxillary_axes_plan test/puzzle_piece_auxillary_axes_plan.test src/puzzle_piece_auxillary_axes_plan.cpp)
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

