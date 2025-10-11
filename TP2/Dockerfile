FROM osrf/ros:jazzy-desktop

RUN apt-get update && \
    apt install ros-jazzy-navigation2 && \
    apt install ros-jazzy-nav2-bringup && \
    apt install ros-jazzy-turtlebot3* && \
    export TURTLEBOT3_MODEL=waffle && \
    export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/opt/ros/jazzy/share/turtlebot3_gazebo/models

