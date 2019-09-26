#!/bin/bash

NPROC=${NPROC:-$(nproc)}

if [[ "$(which nvidia-smi)" == "" ]]; then
    bin/rest-server --cpu-threads $(nproc) $@
else
    bin/rest-server $@
fi
