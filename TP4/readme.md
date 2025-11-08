# Instalacion Inicial
## Instalacion de anaconda en ubuntu

sudo apt-get update
cd /tmp
apt-get install wget
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
bash Anaconda3-2022.05-Linux-x86_64.sh
eval "$(/home/bruno/anaconda3/bin/conda shell.bash hook)"





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





