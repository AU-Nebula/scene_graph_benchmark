ARG CUDA="10.1"
ARG CUDNN="7"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu18.04

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

WORKDIR /sgb

# install basics & create virtual environment

RUN apt-get update -y \
 && apt-get install -y apt-utils git curl ca-certificates bzip2 cmake tree htop bmon iotop g++ \
 && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev libyaml-dev nano wget graphviz graphviz-dev \
 && apt-get install -y ffmpeg libgl1-mesa-glx software-properties-common python3 python3-pip python3-venv

RUN pip3 install --upgrade pip 
RUN apt-get install  -y libjpeg8-dev zlib1g-dev
RUN pip3 install --ignore-installed pillow

RUN pip install requests ninja cython yacs>=0.1.8 numpy>=1.19.5 cython matplotlib opencv-python \
 protobuf tensorboardx pymongo sklearn boto3 scikit-image cityscapesscripts pydot pygraphviz graphviz
RUN pip install azureml-defaults>=1.0.45 azureml.core inference-schema opencv-python timm einops 

# install requirements

# COPY requirements.txt .
# RUN pip3 install -r requirements.txt

RUN pip3 install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html 

RUN pip3 --no-cache-dir install --force-reinstall -I pyyaml
RUN pip3 install opencv-python pycocotools

# install latest PyTorch 1.7.1

# RUN pip3 freeze > requirements.txt

# install PyTorch Detection
ARG FORCE_CUDA="1"
ENV FORCE_CUDA=${FORCE_CUDA}

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

#COPY maskrcnn_benchmark setup.py scene_graph_benchmark tools tests sgg_configs ./

#RUN python setup.py build develop
