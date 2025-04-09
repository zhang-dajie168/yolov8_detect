from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():	

    return LaunchDescription([
        Node(
            package='yolov8_detect',
            executable='yolov8_track_depth_node',
            name='tracker_node',
            parameters=[{
                'model_path':'/root/ros2_ws/install/yolov8_detect/share/yolov8_detect/model/yolov8.rknn'}]
        )
    ])
