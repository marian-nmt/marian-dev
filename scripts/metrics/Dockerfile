# syntax = docker/dockerfile:experimental
FROM mcr.microsoft.com/azureml/minimal-ubuntu22.04-py39-cuda11.8-gpu-inference:20240205.v2
# use this if microsoft image is not accessible;
#FROM nvidia/cuda:11.1.1-devel-ubuntu20.04
LABEL description="Marian image - Ubuntu 22.04"

# required for microsoft cr image
USER root

ARG DEBIAN_FRONTEND=noninteractive
ARG MARIAN_REPO="https://github.com/marian-nmt/marian-dev"
ARG MARIAN_BRANCH=master

RUN apt-get update \
    && apt-get install -y wget apt-utils python3-pip git cmake build-essential \
        intel-mkl openmpi-common openmpi-bin libopenmpi-dev  pkg-config \
    && apt-get clean

RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# install unbabel-comet (requires pytorch) and bleurt (requires tensorflow and cudnn)
# note: unabel-comet 2.x is broken use 1.x. requires numpy < 1.24
    #&& pip install torch==1.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html \
RUN  pip install --upgrade pip \
    && pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118 \
    && pip install sacrebleu unbabel-comet==2.2.1 git+https://github.com/google-research/bleurt.git \
    && rm -rf  ~/.cache/pip/

# Install sentencepiece
RUN pip3 uninstall -y sentencepiece && \
    mkdir -p src && \
    cd src && \
    git clone https://github.com/marian-nmt/sentencepiece && \
    cd sentencepiece && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    make -j install && \
    cd ../python && \
    python3 setup.py install && \
    cd ../../.. && \
    rm -rf src

# add source repo (sans the .dockerignore files)
ADD . /marian-dev
# RUN git clone -b ${MARIAN_BRANCH} ${MARIAN_REPO} /marian \

RUN --mount=type=cache,target=/marian-dev/build mkdir -p /marian-dev/build && cd /marian-dev/build \
    && cmake .. -DUSE_MPI=on -DUSE_STATIC_LIBS=on -DCOMPILE_PASCAL=on -DCOMPILE_VOLTA=on -DCOMPILE_AMPERE=off -DBUILD_ARCH=x86-64 -DCOMPILE_AVX512=off -DPYMARIAN=on \
    && make -j && cp -v marian spm_encode spm_decode /usr/bin/ \
    && pip install -v pymarian-*.whl
