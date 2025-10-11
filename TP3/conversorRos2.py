import os

rosbags = ["cam_april", "imu_april", "cam_checkerboard"]

for bag in rosbags:
    os.system(f"rosbags-convert --src ./archivos/rosbags/ros1/{bag}.bag --dst ./archivos/rosbags/ros2/{bag}")
