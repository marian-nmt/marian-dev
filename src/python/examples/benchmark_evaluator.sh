#!/usr/bin/env bash
set -euo pipefail

mydir=$(dirname $0)


tmpdir=$mydir/tmp.testdata
mkdir -p $tmpdir

# CPU only
MARIAN_ARGS="--cpu-threads 12"
n_recs=5

# IF GPU is available
MARIAN_ARGS="--devices 0 1 2 3"
n_recs=20

log(){ echo "$@" >&2; }
BACKENDS=()

pip list | grep pymarian >& /dev/null && { log "pymarian found"; BACKENDS+=(pymarian); } || { log "pymarian not found"; }
which marian >&2 && { log "marian CLI bin found"; BACKENDS+=(subprocess); } || { log "marian CLI bin not found"; }
[[ ${#BACKENDS[@]} -eq 0 ]] && { log "No Marian backend found"; exit 1; }


LANGS=(en-de en-ru)
for langs in ${LANGS[@]}; do
   prefix=$tmpdir/$langs
   teset=wmt21/systems
   sysname=Online-B
   [[ -s $prefix.src ]] || sacrebleu -t $teset -l $langs --echo src > $prefix.src
   [[ -s $prefix.ref ]] || sacrebleu -t $teset -l $langs --echo ref > $prefix.ref
   [[ -s $prefix.mt ]] || sacrebleu -t $teset -l $langs --echo $sysname > $prefix.mt
done


for langs in ${LANGS[@]}; do
   model="cometoid22-wmt22"
   for backed in ${BACKENDS[@]}; do
      paste $prefix.{src,mt} | head -$n_recs | python -m pymarian.evaluate -m $model --stdin $MARIAN_ARGS --backend $backed || log "$backed $model exited with code $?"
   done

   model="comet20-da"
   for backed in ${BACKENDS[@]}; do
      paste $prefix.{src,mt,ref} | head -$n_recs | python -m pymarian.evaluate -m $model --stdin $MARIAN_ARGS --backend $backed || log "$backed $model exited with code $?"
   done

   model="bleurt20"
   for backed in ${BACKENDS[@]}; do
      paste $prefix.{ref,mt} | head -$n_recs | python -m pymarian.evaluate -m $model --stdin $MARIAN_ARGS --backend $backed || log "$backed $model exited with code $?"
   done
done
