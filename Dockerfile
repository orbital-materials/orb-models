FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

## Install system requirements
RUN apt-get update && \
    apt-get install -y \
        ca-certificates \
        wget \
        git \
        sudo \
        gcc \
        g++

# Help Numba find lubcudart.so
ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
RUN ln -s \
    /opt/conda/lib/python3.11/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12 \
    /opt/conda/lib/python3.11/site-packages/nvidia/cuda_runtime/lib/libcudart.so

## Install Python requirements
RUN pip install orb-models && \
    pip install "cuml-cu12==25.2.*"