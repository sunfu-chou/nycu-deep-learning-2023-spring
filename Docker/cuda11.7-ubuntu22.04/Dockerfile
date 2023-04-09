FROM nvidia/cuda:11.7.0-cudnn8-devel-ubuntu22.04

ENV USER user

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        sudo \
        vim \
        less \
        unzip \
        build-essential \
        gdb \
        ca-certificates \
        ccache \
        cmake \
        curl \
        git \
        libjpeg-dev \
        libpng-dev \
        python3.8 \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install \
        torch \
        torchvision \
        torchaudio \
        matplotlib \
        jupyterlab \
        black \
        tk \
        pandas

RUN useradd -rm -d /home/${USER} -s /bin/bash -g root -G sudo -u 1000 ${USER}
RUN echo "${USER} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
