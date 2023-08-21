#!/usr/bin/env bash
set -eu

MYDIR=$(realpath $(dirname ${BASH_SOURCE[0]}))


METRICS_CACHE=$HOME/.cache/marian/metrics

log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $@" >&2
}

which marian > /dev/null || {
    log "marian not found in PATH. Please add marian binary to \$PATH and rerun"
    exit 2
}

metric_name=
src_file=
ref_file=
hyp_file=
is_seg=
debug_mode=
batch_size=32
pool_size=10
max_length=256
devices=0
workspace=-4000

usage() {
    log " ${BASH_SOURCE##*/} -n METRIC -m HYP [-s SRC] [-r REF] [-d DEVICES] [--seg] [--debug] [-h|--help]

Args:
    -n|--name|--metric NAME  Metric name; required. See below for details.
    -m|--mt|--hyp FILE       MT hypothesis, required for all metrics.
    -s|--src FILE     Source file, required for source based metrics.
    -r|--ref FILE     Reference file, required for reference based metrics.
    -d|--devices DEV  IDs of GPU devices to use. Use quoted string to pass multiple values. Default: '$devices'
    --seg             Output segment-level scores. Default: print only the corpus-level score (mean of segment scores)
    --debug           Enable verbose mode (default is quiet)
    -h|--help         Print this help message

Metric name (-n|--name) shuld be a subdir name under $METRICS_CACHE.
The metric name should have a suffix (-src|-qe|-ref|-src+ref) indicating the type of metric:
    *-src|*-qe   Source-based metric and requires --src arg, e.g., comet20-src or comet20-da-qe
    *-ref        Reference-based metric and requires --ref arg, e.g., bleurt20-ref
    *-src+ref    Both source and reference based and requires --src and --ref args e.g., comet20-src+ref
"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--src)       src_file=$2; shift 2;;
        -r|--ref)       ref_file=$2; shift 2;;
        -m|--mt|--hyp)  hyp_file=$2; shift 2;;
        -n|--name|--metric) metric_name=$2; shift 2;;
        -d|--devices)   devices=$2; shift 2;;
        --seg)          is_seg=1; shift 1;;
        --debug)        debug_mode=1; shift 1;;
        -h|--help)      usage; exit 0;;
        *) log "ERROR: unknown option $1"; usage; exit 1;;
    esac
done

[[ -n $metric_name ]] || { log "ERROR: metric_name=$metric_name name not provided"; usage; exit 1; }
[[ -e $hyp_file ]] || { log "ERROR: hyp file not provided"; usage; exit 1; }

metric_dir=$METRICS_CACHE/$metric_name
checkpoint=$(echo $metric_dir/*model.npz)  # file model.npz or <blah>.model.npz
vocab=$(echo $metric_dir/*vocab.spm)
[[ -f $checkpoint && -f $vocab ]] || {
    log "ERROR: metric $metric_name is not valid. See ls $METRICS_CACHE/$metric_name/{*model.npz,*vocab.spm}"
    exit 1
}

# args common to all models
cmd="marian evaluate -w -4000"
[[ -n $devices ]] && cmd+=" -d $devices"
[[ -n $debug_mode ]] || cmd+=" --quiet"
cmd+=" -m $checkpoint --max-length $max_length --max-length-crop --mini-batch $batch_size --maxi-batch $pool_size -t stdin --tsv"
input=  # to be filled later


check_file(){
    local name=$1
    local file=$2
    [[ -e $file ]] || { log "ERROR: $name file $file does not exist"; exit 1; }
    [[ -s $file ]] || { log "ERROR: $name file $file is empty"; exit 1; }
}

metric_type=${metric_name##*-}   # suffix expected: src, ref, src+ref
case $metric_type in
    src|qe)
        # two sequences: src, hyp
        check_file src $src_file
        cmd+=" --like comet-qe -v $vocab $vocab"
        input="paste $src_file $hyp_file"
        ;;
    ref)
        check_file ref $ref_file
        # two sequences: ref, hyp
        cmd+=" --like bleurt -v $vocab $vocab"
        input="paste $ref_file $hyp_file"
        ;;
    src+ref)
        # three sequences: src, hyp, ref;  three vocabularies
        check_file src $src_file
        check_file ref $ref_file
        cmd+=" --like comet -v $vocab $vocab $vocab"
        input="paste $src_file $hyp_file $ref_file"
        ;;
    *)
        log "ERROR: $metric_name is not valid. Valid metrics have suffix '-{src|qe|ref|src+ref}'"
        exit 3
        ;;
esac

if [[ -z $is_seg ]]; then
    cmd+=" --average only";
fi
pipeline="$input | $cmd | cut -f1 -d' '"

# mean (default) or segment-level scores

log "Running: $pipeline"
eval $pipeline
