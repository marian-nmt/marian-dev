#!/bin/bash

gpus=${1:-8}
scriptPath=$(dirname $0)
root=$scriptPath/../../../../.
marian=$root/build/marian
bleurt=$root/scripts/bleurt
devices=$(seq 0 $(($gpus-1)))

# we reverse the input here since the scorer expects "hyp<tab>ref" but we output pseudo-references first
perl -F'\t' -ane 'chomp(@F); print "$F[1]\t$F[0]\n"' \
| $marian evaluate -m $bleurt/bleurt-20.npz -v $bleurt/bleurt-vocab.{spm,spm} --like bleurt -d $devices --fp16 --quiet
