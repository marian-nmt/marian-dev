# -*- mode: makefile-gmake; indent-tabs-mode: true; tab-width: 4 -*-
SHELL   = bash
mydir  := $(dir $(lastword ${MAKEFILE_LIST}))

REGISTRY=mariannmt

TAG=latest
# image/build-environment: TAG=$(shell md5sum build-environment/Dockerfile | cut -d' ' -f1)
image/build-environment: IMAGE=build-environment
image/build-environment: FLAGS=${DOCKER_BUILD_ARGS}
image/build-environment:
	docker build -t ${REGISTRY}/${IMAGE}:${TAG} ${IMAGE}

image/marian-compiled: IMAGE=marian-compiled
image/marian-compiled: FLAGS=${DOCKER_BUILD_ARGS}
image/marian-compiled:
	docker build -t ${REGISTRY}/${IMAGE}:${TAG} ${FLAGS} ${IMAGE}

image/marian-runtime: IMAGE=marian-runtime
image/marian-runtime: FLAGS=${DOCKER_BUILD_ARGS}
image/marian-runtime:
	docker build -t ${REGISTRY}/${IMAGE}:${TAG} ${FLAGS} ${IMAGE}

