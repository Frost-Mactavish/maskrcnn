#! /bin/bash
# run the script where the setup.py locates

env_name=$1

conda create -n ${env_name} python=3.8 -y
conda activate ${env_name}

conda install gcc=9.5 gxx=9.5 cudatoolkit-dev=11.3 -y
conda install ipython pip -y
conda install ninja yacs cython matplotlib tqdm -y
conda install tensorboard==2.10.0 numpy==1.21.5 scipy==1.7.3 six==1.16.0 setuptools==59.5.0 -y
conda install pytorch==1.10.0 torchvision==0.11.0 cudatoolkit=11.3 -c pytorch
pip install opencv-python

export INSTALL_DIR=$PWD

cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

cd $INSTALL_DIR
git clone https://github.com/mcordts/cityscapesScripts.git
cd cityscapesScripts/
python setup.py build_ext install

# specify cuda dir for compatibility of torch>=1.5
cd $INSTALL_DIR
cuda_dir="maskrcnn_benchmark/csrc/cuda"
perl -i -pe 's/AT_CHECK/TORCH_CHECK/' $cuda_dir/deform_pool_cuda.cu $cuda_dir/deform_conv_cuda.cu
python setup.py build develop

cd $INSTALL_DIR
rm -r cocoapi/ cityscapesScripts/

unset INSTALL_DIR
