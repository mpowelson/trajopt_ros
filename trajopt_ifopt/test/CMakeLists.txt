find_package(GTest REQUIRED)

macro(add_gtest test_name test_file)
  add_executable(${test_name} ${test_file})
  target_compile_options(${test_name} PRIVATE -Wall -Wextra -Wsuggest-override -Wconversion -Wsign-conversion)
  if(CXX_FEATURE_FOUND EQUAL "-1")
      target_compile_options(${test_name} PUBLIC -std=c++11)
  else()
      target_compile_features(${test_name} PUBLIC cxx_std_11)
  endif()
  target_compile_definitions(${test_name} PRIVATE TRAJOPT_DIR="${CMAKE_SOURCE_DIR}")
  target_link_libraries(${test_name}
      ${PROJECT_NAME}
      ${GTEST_BOTH_LIBRARIES}
      ${Boost_SYSTEM_LIBRARY}
      ${Boost_PROGRAM_OPTIONS_LIBRARY})
  target_include_directories(${test_name} PRIVATE
      "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
      ${GTEST_INCLUDE_DIRS})
  target_include_directories(${test_name} SYSTEM PRIVATE)
  add_test(${test_name} ${test_name})
  add_dependencies(${test_name} ${PACKAGE_LIBRARIES})
  add_dependencies(run_tests ${test_name})
endmacro()

add_gtest(${PROJECT_NAME}_variable_sets_unit variable_sets_unit.cpp)

