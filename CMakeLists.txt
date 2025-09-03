cmake_minimum_required(VERSION 3.31)
project(AmmoDetector)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_executable(AmmoDetector main.cpp)
target_include_directories(AmmoDetector PUBLIC ${OpenCV_INCLUDE_DIRS} include/tiny_dnn)
target_link_libraries(AmmoDetector ${OpenCV_LIBS})
