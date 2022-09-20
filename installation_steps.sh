#!/bin/bash
set -e

ForceCuda=${1:-1}

# Download Visual Genome metadata
sh scripts/download_VG.sh
# Download pre-trained model
sh scripts/download_pretrained.sh

# Download NVIDIA Docker image
docker pull nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
# Build Docker image
docker build -t au/sgb:10.1-cudnn7-devel-ubuntu18.04-pip3 . --build-arg FORCE_CUDA=${ForceCuda}