FROM mcr.microsoft.com/azureml/minimal-ubuntu20.04-py38-cuda11.6.2-gpu-inference:20231102.v2
# use this if microsoft image is not accessible;
#FROM nvidia/cuda:11.1.1-devel-ubuntu20.04
LABEL description="Marian image - Ubuntu 20.04"

ARG DEBIAN_FRONTEND=noninteractive
ARG NCPU=24
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
RUN  pip install --upgrade pip \
    && pip install torch==1.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html \
    && pip install sacrebleu unbabel-comet==1.1.3 numpy==1.23.5 nvidia-cudnn-cu11==8.6.0.163 git+https://github.com/google-research/bleurt.git \
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

RUN git clone -b ${MARIAN_BRANCH} ${MARIAN_REPO} /marian \
    && mkdir /marian/build && cd /marian/build \
    && cmake .. -DUSE_MPI=on -DUSE_STATIC_LIBS=off -DCOMPILE_PASCAL=on -DCOMPILE_VOLTA=on -DCOMPILE_AMPERE=off -DBUILD_ARCH=x86-64 -DCOMPILE_AVX512=off \
    && make -j $NCPU && cp -v marian spm_encode spm_decode /usr/bin/ \
