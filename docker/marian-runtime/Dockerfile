ARG MARIAN_COMPILED=mariannmt/marian-compiled
ARG BASE_IMAGE=nvidia/cuda:9.2-cudnn7-devel-ubuntu16.04
FROM ${MARIAN_COMPILED} as compiled
FROM ${BASE_IMAGE}

LABEL MAINTAINER="Ulrich Germann <ugermann@inf.ed.ac.uk>"

WORKDIR /opt/app/marian
COPY --from=compiled \
     /marian-dev/build/rest-server  \
     /marian-dev/build/websocket-server  \
     /marian-dev/build/amqp-worker \
     /marian-dev/build/marian-decoder \
     /opt/app/marian/bin/

COPY --from=compiled \
     /marian-dev/src/server/rest \
     /opt/app/marian/rest



