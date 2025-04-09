cmake_minimum_required(VERSION 3.10)

project(yolov8_detect)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


#  设置库架构
set(LIB_ARCH "aarch64")
set(DEVICE_NAME "RK3588")

# 寻找Eigen3
set(EIGEN_VERSION 3.3)
find_package(Eigen3 ${EIGEN_VERSION} QUIET)


#  rknn_api 文件夹路径
set(RKNN_API_PATH ${CMAKE_CURRENT_SOURCE_DIR}/librknn_api)
#  rknn_api include 路径
set(RKNN_API_INCLUDE_PATH ${RKNN_API_PATH}/include)
#  rknn_api lib 路径
set(RKNN_API_LIB_PATH ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)

# 寻找OpenCV库，使用自定义的OpenCV_DIR
find_package(OpenCV REQUIRED)

# 输出OpenCV信息
message(STATUS "include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "${OpenCV_VERSION}")

# 查找依赖
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(CURL REQUIRED)
find_package(message_filters REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
link_directories( ${CMAKE_CURRENT_SOURCE_DIR}/lib)


# 用来搜索头文件的目录
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${RKNN_API_INCLUDE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
    ${YOLO_API_INCLUDE_PATH}
)


# bytetrack_lib
add_library(bytetrack_lib SHARED
  src/bytetrack/BYTETracker.cpp
  src/bytetrack/kalmanFilter.cpp
  src/bytetrack/lapjv.cpp
  src/bytetrack/STrack.cpp
  src/bytetrack/utils.cpp
  )

target_link_libraries(bytetrack_lib Eigen3::Eigen)

# 测试yolov8_video_track
add_executable(yolov8_track_depth_node
src/yolov8_track_depth_node.cpp
src/utils/yolov8.cc
src/utils/file_utils.c
src/utils/image_drawing.c
src/utils/image_utils.c
src/utils/postprocess.cc
src/utils/encode_frame.cc
)

ament_target_dependencies(yolov8_track_depth_node
   rclcpp
   sensor_msgs
   cv_bridge
   vision_msgs
   image_transport
   geometry_msgs
   message_filters
)
target_link_libraries(yolov8_track_depth_node 
 m
  rknnrt
  turbojpeg
  rga
  bytetrack_lib
  ${CURL_LIBRARIES}
  ${OpenCV_LIBS}
  )


install(TARGETS yolov8_track_depth_node
bytetrack_lib
  DESTINATION lib/${PROJECT_NAME}
)
#install(DIRECTORY include/
#  DESTINATION include
#)

install(TARGETS yolov8_track_depth_node
  DESTINATION lib/${PROJECT_NAME})
 
install(DIRECTORY include lib
  DESTINATION share/${PROJECT_NAME}
)
 install(DIRECTORY launch
      DESTINATION share/${PROJECT_NAME}/launch
 )
 
 install(DIRECTORY model/
  DESTINATION share/${PROJECT_NAME}/model
)
 
 
  
ament_package()
