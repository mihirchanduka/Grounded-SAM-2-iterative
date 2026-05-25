FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

# Arguments to build Docker Image using CUDA
ARG USE_CUDA=1
ARG TORCH_ARCH="7.5 8.0 8.6 8.9 9.0 10.0 12.0+PTX"

ENV AM_I_DOCKER=True
ENV BUILD_WITH_CUDA="${USE_CUDA}"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_ARCH}"
ENV CUDA_HOME=/usr/local/cuda-12.8/
# Ensure CUDA is correctly set up
ENV PATH=/usr/local/cuda-12.8/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH}

# Install required packages and specific gcc/g++
RUN apt-get update && apt-get install --no-install-recommends wget ffmpeg=7:* \
    libsm6=2:* libxext6=2:* git=1:* nano vim=2:* ninja-build gcc-10 g++-10 -y \
    && apt-get clean && apt-get autoremove && rm -rf /var/lib/apt/lists/*

ENV CC=gcc-10
ENV CXX=g++-10

# Install essential Python packages
RUN python -m pip install --upgrade pip setuptools wheel numpy \
    opencv-python transformers supervision pycocotools addict yapf timm pyzmq

RUN mkdir -p /home/appuser/Grounded-SAM-2
COPY . /home/appuser/Grounded-SAM-2/
COPY ./patched.ms_deform_attn_cuda.cu /home/appuser/Grounded-SAM-2/grounding_dino/groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn_cuda.cu
WORKDIR /home/appuser/Grounded-SAM-2

# Install segment_anything package in editable mode
RUN python -m pip install -e .

# Install grounding dino 
RUN python -m pip install --no-build-isolation -e grounding_dino

RUN ls -la
WORKDIR /home/appuser/Grounded-SAM-2/checkpoints
RUN ls -la
RUN bash download_ckpts.sh
WORKDIR /home/appuser/Grounded-SAM-2/gdino_checkpoints
RUN bash download_ckpts.sh

WORKDIR /home/appuser/Grounded-SAM-2
#RUN python huggingface_downloads.py
