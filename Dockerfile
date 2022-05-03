ARG CUDA="10.1"
ARG CUDNN="7"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu18.04

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

WORKDIR /sgb

# Temporary FIX
# https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212771/8
# TODO: Around May29th check if re-downloading the cuda image without this works
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub 2
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub 2

# install basics & create virtual environment

RUN apt-get update -y \
 && apt-get install -y apt-utils git curl ca-certificates bzip2 cmake tree htop bmon iotop g++ \
 && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev libyaml-dev nano wget graphviz graphviz-dev \
 && apt-get install -y ffmpeg libgl1-mesa-glx software-properties-common python3 python3-pip python3-venv

RUN pip3 install --upgrade pip 
RUN apt-get install  -y libjpeg8-dev zlib1g-dev
RUN pip3 install --ignore-installed pillow

# install requirements

COPY requirements.txt .
RUN pip3 install -r requirements.txt

# install latest PyTorch 1.7.1

RUN pip3 install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html 

# install PyTorch Detection
ARG FORCE_CUDA="1"
ENV FORCE_CUDA=${FORCE_CUDA}

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
