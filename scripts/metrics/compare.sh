#!/usr/bin/env bash

# This script compares the scores produced by
# original implementation (unbabel-score or BLEURT) and Marian NMT (pymarian-eval).


MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
OUT_DIR=$MYDIR/workspace
REGEN_ORIG=0   # 1 : clear and regenerate original scores. 0: reuse previous scores
REGEN_MARIAN=0  # 1 : to clear and regenerate marian scores (recommended).  0:  reuse / resume from previous scores

DEVICES=0
cd $MYDIR
export CUDA_VISIBLE_DEVICES=0

# add source to python path to test changes before installing
# export PYTHONPATH=$(cd $MYDIR/../../src/python && pwd)

log() {
    echo -e "\e[1;32m[$(date '+%Y-%m-%d %H:%M:%S')]\e[0m $@" >&2
}

for tool in comet-score pymarian-eval; do
    which $tool > /dev/null || {
        log "ERROR: $tool not found in PATH"
        exit 1
    }
done


METRIC_NAMES=$(cat $MYDIR/known-models.txt | grep -v '^#' | awk '{print $1}')
# exclude xxl, they require more memory
METRIC_NAMES=$(grep -v -i '\-xxl\|xcomet' <<< $METRIC_NAMES)

get_sacrebleu_names(){
    set -eu
    # using sacrebleu to get the list of systems
    testset=wmt21/systems
    while read line; do
        pair=$(cut -f1 -d':' <<< $line)
        refs=()
        mts=()
        while read name; do
            # skip if name starts with $pair or src or docidq
            if [[ $name == $pair* || $name == src || $name == docid || $name == origlang ]]; then
                continue
            fi
            if [[ $name == ref* ]]; then
                refs+=($name)
            else
                mts+=($name)
            fi
        done < <(sed 's/,//g;s/ /\n/g' <<< $line)

        # flatten: ref x mt
        for ref in ${refs[@]}; do
            for mt in ${mts[@]}; do
                echo -e "$testset\t$pair\t$ref\t$mt"
                break  # limit to one per lang pair
            done
            break  # limit to one per lang pair
        done
    done < <(sacrebleu -t $testset --list)
}

unbabel_score(){
    set -eu
    local metric=$1
    local prefix=$2
    log "Running $metric"
    local batch_size=64
    comet-score --batch_size $batch_size --model $metric -s $prefix.src -r $prefix.ref -t $prefix.mt \
        | awk -F '[:\t]' 'NF==4{print $NF}'
}


bleurt_score() {
    set -eu
    local metric_name=$1
    local prefix=$2
    [[ $metric_name == "BLEURT-20" ]] || {
        log "ERROR: BLEURT-20 is the only supported metric; given: $metric_name"
        exit 1
    }
    local cache_dir=$HOME/.cache/bleurt
    local metric_path=$cache_dir/$metric_name
    [[ -f $metric_path/._OK ]] || {
        log "BLEURT model not found in $HOME/.cache/bleurt .. Downloading"
        mkdir -p $cache_dir
        rm -rf $metric_path.zip   # remove incomplete file
        wget https://storage.googleapis.com/bleurt-oss-21/$metric_name.zip -P $cache_dir \
            && unzip $metric_path.zip -d $cache_dir/ && touch $metric_path/._OK
    }

    # to check if cuda libs are configured and GPU is available
    # python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
    #export LD_LIBRARY_PATH=/usr/local/lib/python3.8/dist-packages/nvidia/cudnn/lib/:$LD_LIBRARY_PATH
    python -m bleurt.score_files --bleurt_checkpoint=$metric_path \
        --candidate_file=$prefix.mt --reference_file=$prefix.ref \
        --bleurt_batch_size 64 2> /dev/null
}


MAX_TESTS=10
MAX_LINES=100  # in each testset
mkdir -p $OUT_DIR

while IFS=$'\t' read tset pair ref mt; do
    data=$(sacrebleu -t $tset -l $pair --echo src ref $mt)
    prefix=$OUT_DIR/${tset//\//-}.$pair.$MAX_LINES

    [[ -s $prefix.src ]] || cut -f1 <<< "$data" | head -n $MAX_LINES > $prefix.src
    [[ -s $prefix.ref ]] || cut -f2 <<< "$data" | head -n $MAX_LINES > $prefix.ref
    [[ -s $prefix.mt ]] || cut -f3 <<< "$data" | head -n $MAX_LINES > $prefix.mt

    report_file=$prefix.report.txt
    echo "####$(date '+%Y-%m-%d %H:%M:%S') :: $(pymarian-eval -V) :: Avg diffs" | tee -a $report_file

    for mn in ${METRIC_NAMES[@]}; do
        log "Comparing >> $mn << on $tset $pair $ref $mt"
        metric_id=$(basename $mn | tr '[:upper:]' '[:lower:]')
        score_pref=$prefix.$metric_id
        orig_file=$score_pref.orig
        if [[ ! -s $orig_file || $REGEN_ORIG -eq 1 ]]; then
            rm -f $score_pref  # cleanup
            log "Generating original scores for $mn :: $prefix"
            if [[ $mn =~ BLEURT* ]]; then
                bleurt_score $mn $prefix > $orig_file
            else
                unbabel_score $mn $prefix 2> /dev/null > $orig_file
            fi
        fi

        out_file=$score_pref.pymarian
        if [[ ! -s $out_file || $REGEN_MARIAN -eq 1 ]]; then
            rm -f $out_file $out_file.log  # cleanup
            log "Generating Marian scores for $mn :: $prefix"
            pymarian-eval -d $DEVICES -m $(basename $mn) -s $prefix.src -r $prefix.ref -t $prefix.mt -a skip --fp16 --debug > $out_file 2> $out_file.log || {
                log "ERROR: Failed to generate scores for $mn"
                cat $out_file.log
                continue
            }
        fi

        # compute diffs
        paste $out_file $orig_file \
            | awk -F '\t' -v OFS='\t' -v mn=$mn -v of=$out_file.diff 'BEGIN {tot=0.0}
                {$2 = +sprintf("%.4f", $2); diff=sqrt(($1-$2)^2); tot+=diff; print diff, $0 > of}
                END {printf "%s\t%f\n", mn, tot/NR}' | tee -a $report_file
    done
done <  <(get_sacrebleu_names | head -n $MAX_TESTS)

cat $OUT_DIR/*.report.txt #| column -t
