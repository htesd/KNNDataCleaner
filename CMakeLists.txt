cmake_minimum_required(VERSION 3.16)
project(KNN)
find_package(OpenCV 4 REQUIRED)
set(CMAKE_CXX_STANDARD 14)

add_executable(KNN main.cpp)
target_link_libraries(KNN ${OpenCV_LIBS})