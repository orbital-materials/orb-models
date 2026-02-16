FROM pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

## Install system requirements
RUN apt-get update && \
    apt-get install -y \
        ca-certificates \
        wget \
        git \
        sudo \
        gcc \
        g++ \ 
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

## Install Python requirements
# For details on --break-system-packages, see: https://veronneau.org/python-311-pip-and-breaking-system-packages.html 
RUN pip install --break-system-packages orb-models && \
    pip install --break-system-packages "cuml-cu12==25.2.*"
