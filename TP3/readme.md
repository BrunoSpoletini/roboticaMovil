# Instalacion Inicial
## Instalacion del distro jazzy
Para instalar el distro es necesario correr los siguientes comandos
```
sudo apt update
sudo apt install ros-jazzy-camera-calibration-parsers
sudo apt install ros-jazzy-camera-info-manager
sudo apt install ros-jazzy-launch-testing-ament-cmake
```
## Instalacion Image Pipeline
La instancion depende del tipo de config de git que tengas, puede ser
```
git clone -b jazzy git@github.com:ros-perception/image_pipeline.git
```
Tambien se puede usar
```
gh repo clone ros-perception/image_pipeline
cd image_pipeline
git checkout jazzy
cd ..
```
Luego es necesario compilar y sourcear el image_pipeline
```
colcon build --symlink-install
source install/setup.bash
```
# Ejercicio 1
Primero es recomendado entrar al directorio del Trabajo Praactico, ya que ahi se encutran los archivos necesario
```
cd TP3
```
Iniciar la camara en loop desde el bag de ros2
```
ros2 bag play <calibration_rosbag>
```
Donde <calibration_rosbag> es el path a la rosbag de calibracion
\
\
Iniciar la calibracion 
```
ros2 run camera_calibration cameracalibrator --size 7x6 --square 0.06 --no-service-check --ros-args --remap left:=/cam0/image_raw --ros-args --remap right:=/cam1/image_raw -p camera:=/my_camera
```
--no-service-check es para que el calibrador no busque el servicio set_camera_info del driver de la camara, ya que estamos publicando la imagen desde un bag, que no tiene esa info
\
\
Ver la camara en rviz (solo para hacer debug)
```
ros2 run rviz2 rviz2
```
# Ejercicio 2
## Instalacion Librerias pip
Para instalar librerias por pip es necesario entrar a una virtual machien
```
source ~/venv_sfm/bin/activate
```
Luego instalamos las librerias con pip
```
pip install -r requirements.txt
```
## Compilacion del Publicador
Primero es necesario compilar el publicador
```
colcon build --packages-select camera_info_publisher
```
Luego hay que hacer el source
```
source install/setup.bash
```
## Publicar
Para publicar es necesario correr el siguiente comando
```
ros2 run camera_info_publisher publisher <input>
```
Donde
* <input> : carpeta con los archivos de entrada
La carpeta <input> debe estructurarse de la sigueinte forma
```
.
└── input/
    ├── calibration_left.yaml     # Archivo de calibracion camara izquierda 
    ├── calibration_right.yaml    # Archivo de calibracion camara derecha
    ├── kalibr_imucam_chain.yaml  # Archivo de calibracion imu-camara   
    ├── poses.txt                 # Poses del ground-thruth
    └── rosbag.db3                # Rosbag de la trayectoria
```
