#!/usr/bin/env bash
MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $MYDIR

IMAGE="marian-dev"

VISIBLE_GPUS="1"   # exlcude 0 for now; run on single GPU

MOUNTS="-v $PWD:$PWD"
for cache in .sacrebleu .cache/{marian,torch,huggingface,bleurt}; do
    MOUNTS+=" -v $HOME/$cache:/root/$cache"
done


cmd="docker run --rm -i $MOUNTS --gpus "\"device=$VISIBLE_GPUS\"" -t $IMAGE"

# uncomment for an interactive shell
# $cmd bash

$cmd $PWD/compare.sh $@
