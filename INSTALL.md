## Installation

### Requirements:
- PyTorch 1.7
- torchvision 0.8.2
- cocoapi 
- yacs>=0.1.8
- numpy>=1.19.5
- matplotlib 3.3.4
- GCC >= 4.9
- OpenCV 4.5.5.64
- CUDA >= 10.1

The complete list of requirements is available in the document *requirements.txt*

### Docker Image (Requires CUDA, Linux only)

Build image with defaults (`CUDA=10.1`, `CUDNN=7`, `FORCE_CUDA=1`):

    nvidia-docker build -t scene_graph_benchmark docker/

Build image with FORCE_CUDA disabled:

    nvidia-docker build -t scene_graph_benchmark --build-arg FORCE_CUDA=0 docker/

Build and run image with built-in jupyter notebook(note that the password is used to log in jupyter notebook):

    nvidia-docker build -t scene_graph_benchmark-jupyter docker/docker-jupyter/
    nvidia-docker run -td -p 8888:8888 -e PASSWORD=<password> -v <host-dir>:<container-dir> scene_graph_benchmark-jupyter
