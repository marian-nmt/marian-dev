#!/usr/bin/env bash
set -euo pipefail

mydir=$(dirname $0)

tools=(marian sacrebleu)
for tool in ${tools[@]}; do
  which $tool >&2 || { echo "$tool not found"; exit 1; }
done


tmpdir=$mydir/tmp.testdata
mkdir -p $tmpdir

log(){ echo "$@" >&2; }

for langs in en-de en-ru; do
   prefix=$tmpdir/$langs
   teset=wmt21/systems
   sysname=Online-B
   [[ -s $prefix.src ]] || sacrebleu -t $teset -l $langs --echo src > $prefix.src
   [[ -s $prefix.ref ]] || sacrebleu -t $teset -l $langs --echo ref > $prefix.ref
   [[ -s $prefix.mt ]] || sacrebleu -t $teset -l $langs --echo $sysname > $prefix.mt

   n_recs=5
   model="cometoid22-wmt22"
   n_threads=12
   paste $prefix.{src,mt} | head -$n_recs | $mydir/evaluator.py -m $model --stdin --cpu-threads $n_threads --backend pymarian || log "pymarian $model exited with code $?"
   paste $prefix.{src,mt} | head -$n_recs | $mydir/evaluator.py -m $model --stdin --cpu-threads $n_threads --backend subprocess || log "subprocess $model exited with code $?"
   
   model="comet20-da"
   paste $prefix.{src,mt,ref} | head -$n_recs | $mydir/evaluator.py -m $model --stdin --cpu-threads $n_threads --backend pymarian || log "pymarian $model exited with code $?"
   paste $prefix.{src,mt,ref} | head -$n_recs | $mydir/evaluator.py -m $model --stdin --cpu-threads $n_threads --backend subprocess || log "subprocess $model exited with code $?"

   model="bleurt20"
   paste $prefix.{ref,mt}  | head -$n_recs | $mydir/evaluator.py -m $model --stdin --cpu-threads $n_threads --backend pymarian || log "pymarian $model exited with code $?"
   paste $prefix.{ref,mt} | head -$n_recs | $mydir/evaluator.py -m $model --stdin --cpu-threads $n_threads --backend subprocess || log "subprocess $model exited with code $?"
done
