FROM mariannmt/build-environment
LABEL MAINTAINER="Ulrich Germann <ugermann@inf.ed.ac.uk>"

RUN git clone https://github.com/marian-nmt/marian-dev \
    && mkdir marian-dev/build

WORKDIR marian-dev/build
RUN git checkout ug-marian-aas
RUN cmake \
       -DBUILD_ARCH=x86-64 \
       -DCMAKE_BUILD_TYPE=Release \
       -DUSE_STATIC_LIBS=on \
       -DUSE_SENTENCEPIECE=on \
       .. 
RUN apt-get install -y libev++-dev
RUN make -j
