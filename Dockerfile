FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

LABEL authors="Colby T. Ford <colby@tuple.xyz>"

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
    # pip install "pynanoflann@git+https://github.com/dwastberg/pynanoflann#egg=af434039ae14bedcbb838a7808924d6689274168"
    pip install git+https://github.com/u1234x1234/pynanoflann.git@0.0.8