
cd ros2_ws/

clocon build --packages-select yolov8_detect

export LD_LIBRARY_PATH=/root/ros2_ws/install/yolov8_detect/lib/yolov8_detect:$LD_LIBRARY_PATH

source install/setup.bash

ros2 launch yolov8_detect yolov8_tracking.launch.py




#另外打开终端
rviz2

add->image->topic->output_image
