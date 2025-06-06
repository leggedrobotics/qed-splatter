FROM ghcr.io/nerfstudio-project/nerfstudio

# Install system dependencies
RUN apt-get update && \
    apt-get install -y git gcc-10 g++-10 nvidia-cuda-toolkit ninja-build python3-pip wget && \
    rm -rf /var/lib/apt/lists/*

# Set environment variables to use gcc-10
ENV CC=gcc-10
ENV CXX=g++-10
ENV CUDA_HOME="/usr/local/cuda"
ENV CMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6+PTX"

# Clone QED-Splatter and install in editable mode
RUN git clone https://github.com/leggedrobotics/qed-splatter.git /apps/qed-splatter
RUN pip install /apps/qed-splatter 

# Register QED-Splatter with nerfstudio
RUN ns-install-cli 
RUN mkdir -p /usr/local/cuda/bin && ln -s /usr/bin/nvcc /usr/local/cuda/bin/nvcc

# Install additional Python dependencies
RUN pip install git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e
RUN pip install git+https://github.com/rahul-goel/fused-ssim@1272e21a282342e89537159e4bad508b19b34157
RUN pip install nerfview pyntcloud

# Pre-download AlexNet pretrained weights to prevent runtime downloading
RUN mkdir -p /root/.cache/torch/hub/checkpoints && \
    wget https://download.pytorch.org/models/alexnet-owt-7be5be79.pth \
         -O /root/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth

# Modify gsplat to handle softpruning
COPY soft_pruning_tweeks/default.py /usr/local/lib/python3.10/dist-packages/gsplat/strategy/default.py
COPY soft_pruning_tweeks/soft_prune_utils /usr/local/lib/python3.10/dist-packages/gsplat/strategy/soft_prune_utils

# Set working directory
WORKDIR /workspace
