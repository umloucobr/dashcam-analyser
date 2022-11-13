cmake_minimum_required (VERSION 3.8)

project ("dashcam-analyser")

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable (dashcam-analyser "dashcam-analyser.cpp" "dashcam-analyser.hpp" "car-detecter.cpp" "car-detecter.hpp" "lane-detector.cpp")
target_link_libraries( dashcam-analyser ${OpenCV_LIBS} )

set_property(TARGET dashcam-analyser PROPERTY CXX_STANDARD 20)