# Instalacion

sudo apt update

sudo apt install ros-jazzy-camera-calibration-parsers

sudo apt install ros-jazzy-camera-info-manager

sudo apt install ros-jazzy-launch-testing-ament-cmake

### Depende del tipo de config de git que tengas:
git clone -b jazzy git@github.com:ros-perception/image_pipeline.git
### ó
gh repo clone ros-perception/image_pipeline
cd image_pipeline
git checkout jazzy
cd ..

### Compilar y sourcear el image_pipeline
colcon build --symlink-install
source install/setup.bash


# Ejercicio 1
## Iniciar la camara en loop desde el bag de ros2
ros2 bag play archivos/rosbags/ros2/cam_checkerboard/cam_checkerboard.db3 --loop

## Iniciar la calibracion 
### --no-service-check es para que el calibrador no busque el servicio set_camera_info del driver de la camara, ya que estamos publicando la imagen desde un bag, que no tiene esa info
ros2 run camera_calibration cameracalibrator --size 7x6 --square 0.06 --no-service-check --ros-args --remap left:=/cam0/image_raw --ros-args --remap right:=/cam1/image_raw -p camera:=/my_camera

# Ver la camara en rviz (solo para hacer debug)
ros2 run rviz2 rviz2

# Ejercicio 2

## Publicar la info de las camaras a un topico:
Creamos un paquete "camera_info_publisher" que contiene al script publisher.py, que publica a un topico la info de las camaras izq y der obtenidas en la calibración.
Compilamos y sourceamos el paquete con
    colcon build --packages-select camera_info_publisher && source install/setup.bash
Ejecutamos el script para publicar la info con
    ros2 run camera_info_publisher publisher


## Iniciar la camara en loop desde el bag remapeando los topicos
ros2 bag play archivos/rosbags/ros2/cam_checkerboard/cam_checkerboard.db3 --remap \
    /cam0/image_raw:=/left/image_raw \
    /cam1/image_raw:=/right/image_raw \
    --loop
    
### Apartado a)
ros2 launch stereo_image_proc stereo_image_proc.launch.py