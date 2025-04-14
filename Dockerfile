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

## Install Python requirements
RUN pip install orb-models && \
    pip install --extra-index-url=https://pypi.nvidia.com "cuml-cu12==25.2.*"