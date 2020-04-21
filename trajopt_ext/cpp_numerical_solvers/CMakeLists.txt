cmake_minimum_required(VERSION 3.5.0)
project(cppoptlib VERSION 1.0.0 LANGUAGES CXX)

find_package(cppoptlib QUIET)
find_package(trajopt_utils REQUIRED)

if (NOT ${cpp_numerical_solvers_FOUND})
  message(WARNING "No valid cpp_numerical_solvers version found. Cloning cpp_numerical_solvers master into build directory")

  include(ExternalProject)

  message(WARNING ${CMAKE_BINARY_DIR})
  ExternalProject_Add(cpp_numerical_solvers
    GIT_REPOSITORY    https://github.com/PatWie/CppNumericalSolvers
    GIT_TAG           master
    SOURCE_DIR        ${CMAKE_BINARY_DIR}/../trajopt_ext/cpp_numerical_solvers-src
    BINARY_DIR        ${CMAKE_BINARY_DIR}/../trajopt_ext/cpp_numerical_solvers-build
    # Disable these since this is a non-CMake header only library
    UPDATE_DISCONNECTED 1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND  ""
    INSTALL_COMMAND  ""
    TEST_COMMAND  ""
  )

  # Add headers as interface target
  add_library(${PROJECT_NAME} INTERFACE)
  trajopt_target_compile_options(${PROJECT_NAME} INTERFACE)
  target_include_directories(${PROJECT_NAME} INTERFACE
      "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
      "$<INSTALL_INTERFACE:include>")

  trajopt_configure_package(${PROJECT_NAME})

  # Mark cpp header files for installation
  install(DIRECTORY ${CMAKE_BINARY_DIR}/../trajopt_ext/cpp_numerical_solvers-src/include/cppoptlib
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
    PATTERN ".svn" EXCLUDE
   )

endif()

install(FILES package.xml DESTINATION share/cpp_numerical_solvers)
