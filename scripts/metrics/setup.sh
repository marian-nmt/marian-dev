#!/usr/bin/env bash
MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $MYDIR

#SSH_KEY=$HOME/.ssh/id_rsa    # for git clone inside docker build
IMAGE=marian-dev
echo "Building docker image $IMAGE"
#DOCKER_BUILDKIT=1 docker build --ssh default=$SSH_KEY . -f Dockerfile -t $IMAGE
DOCKER_BUILDKIT=1 docker build . -f Dockerfile -t $IMAGE


# Optional build args:
# --build-arg MARIAN_COMMIT=master \
# --build-arg MARIAN_REPO=https://github.com/marian-nmt/marian-dev.git \
# --build-arg NCPUS=16
