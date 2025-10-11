FROM osrf/ros:jazzy-desktop

# Install packages in one non-interactive RUN and clean apt caches
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ros-jazzy-navigation2 \
        ros-jazzy-nav2-bringup \
        ros-jazzy-turtlebot3* && \
    rm -rf /var/lib/apt/lists/* && \
    git clone https://github.com/BrunoSpoletini/roboticaMovil.git

ENV TURTLEBOT3_MODEL=waffle

ARG GAZEBO_MODEL_PATH
ENV GAZEBO_MODEL_PATH=/opt/ros/jazzy/share/turtlebot3_gazebo/models:${GAZEBO_MODEL_PATH}

# EntryPoint que “sourcea” ROS y luego ejecuta el comando
COPY ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh
ENTRYPOINT ["/ros_entrypoint.sh"]

# Arranca en bash por defecto (podés sobreescribir con `docker run ... <cmd>`)
CMD ["bash"]