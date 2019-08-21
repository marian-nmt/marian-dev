# -*- mode: makefile-gmake; indent-tabs-mode: true; tab-width: 4 -*-
DOCKER_IMAGE=marian-nmt/build-environment:$(shell git rev-parse --short HEAD -- build-environment/Dockerfile)

image:
	docker build -t ${DOCKER_IMAGE} .

