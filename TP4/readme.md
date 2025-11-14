# Instalacion Inicial
## Instalacion de anaconda en ubuntu
Para instalar anaconda en ubuntu es necesario correr los siguientes comandos
```
sudo apt-get update
cd /tmp
apt-get install wget
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
bash Anaconda3-2022.05-Linux-x86_64.sh
eval "$(/home/bruno/anaconda3/bin/conda shell.bash hook)"
```
## Instalar gtsam
Para instalar gtsam usando anaconda es necesario correr los siguientes comandos
```
conda create -n gtsam_env python=3.12
conda activate gtsam_env
conda install -c conda-forge cmake eigen pybind11 boost numpy matplotlib python-graphviz conda-forge::plotly conda-forge::pandas conda-forge::nbformat
git clone https://github.com/borglab/gtsam.git
cd gtsam
mkdir build && cd build
conda install -r ../python/requirements.txt
cmake .. -DGTSAM_BUILD_PYTHON=1 -DGTSAM_PYTHON_VERSION=3.12
make -j2
cmake -DGTSAM_BUILD_PYTHON_STUBS=OFF ..
make python-install
```
# Ejecutar el programa
Para ejecutar el programa, correr el siguiente comando
```
python3 script.py --g2o <g2o-input> --g2o3d <g20-3d-input> --output <outdir> --mode <mode>
```
Donde
- <g2o-input> : es el archivo con del dataset 2D
- <g20-3d-input> : es el archivo con el dataset 3D
- <outdir> : es el directorio donde guardar el output del script
- <mode> : es el modo ejecucion, el mismo puede ser:
  - 2d : utiliza solo el dataset 2D
  - 3d : utiliza solo el dataset 3D
  - both : utiliza ambos datasets


