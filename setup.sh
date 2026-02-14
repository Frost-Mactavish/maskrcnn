#! /bin/bash
# run the script where the setup.py locates with source command
# e.g. source scripts/docker/maskrcnn_setup.sh maskrcnn_env

env_name=$1

conda create -n ${env_name} python=3.8 -y
conda activate ${env_name}

conda install gcc=9.5 gxx=9.5 cudatoolkit-dev=11.1 -c conda-forge -y
conda install scipy==1.7.3 six==1.16.0 setuptools==59.5.0 tensorboard==2.10.0 -c conda-forge -y
conda install ipython pip ninja yacs cython matplotlib tqdm pycocotools -c conda-forge -y
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install opencv-python

python setup.py build develop
python setup.py clean --all