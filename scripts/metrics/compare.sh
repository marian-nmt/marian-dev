#!/usr/bin/env bash
MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export PATH=$MYDIR:$PATH

log() {
    echo -e "\e[1;32m[$(date '+%Y-%m-%d %H:%M:%S')]\e[0m $@" >&2
}

get_sacrebleu_names(){
    # using sacrebleu to get the list of systems
    testset=wmt21/systems
    while read line; do
        pair=$(cut -f1 -d':' <<< $line)
        refs=()
        mts=()
        while read name; do
            # skip if name starts with $pair or src or docid
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
            done
        done
    done < <(sacrebleu -t $testset --list)
}

unbabel_score(){
    local metric=$1
    local prefix=$2
    log "Running $metric"
    local batch_size=64
    comet-score --batch_size $batch_size --model $metric -s $prefix.src -r $prefix.ref -t $prefix.mt \
        | awk -F '[:\t]' 'NF==4{print $NF}'
}


bleurt_score() {
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
    export LD_LIBRARY_PATH=/usr/local/lib/python3.8/dist-packages/nvidia/cudnn/lib/:$LD_LIBRARY_PATH
    python -m bleurt.score_files --bleurt_checkpoint=$metric_path \
        --candidate_file=$prefix.mt --reference_file=$prefix.ref \
        --bleurt_batch_size 64 2> /dev/null
}

marian_score() {
    local metric=$1
    local prefix=$2
    case $metric in
        wmt20-comet-qe-da)  metric="comet20-da-src" ;;
        wmt20-comet-da)     metric="comet20-da-src+ref" ;;
        BLEURT-20)          metric="bleurt20-ref" ;;
        *) echo "Unknown metric $metric";  exit 1;;
    esac
    marian-score.sh -d '0' -n $metric --src $prefix.src --ref $prefix.ref --mt $prefix.mt --seg
}


main() {
    cd $MYDIR
    local metric_names=(BLEURT-20 wmt20-comet-da wmt20-comet-qe-da)
    export CUDA_VISIBLE_DEVICES=0
    local max_tests=10
    local max_lines=100  # in each testset
    while IFS=$'\t' read tset pair ref mt; do
        for mn in ${metric_names[@]}; do
            log "Comparing >> $mn << on $tset $pair $ref $mt"
            local data=$(sacrebleu -t $tset -l $pair --echo src ref $mt)
            local tmp_pref=tmp.testset
            rm -rf $tmp_pref.{src,ref,mt}
            cut -f1 <<< "$data" | head -n $max_lines > $tmp_pref.src
            cut -f2 <<< "$data" | head -n $max_lines > $tmp_pref.ref
            cut -f3 <<< "$data" | head -n $max_lines > $tmp_pref.mt
            if [[ $mn =~ BLEURT* ]]; then
                local orig_out=$(bleurt_score $mn $tmp_pref)
            else
                local orig_out=$(unbabel_score $mn $tmp_pref 2> /dev/null)
            fi
            local marian_out=$(marian_score $mn $tmp_pref)
            paste <(echo "$marian_out") <(echo "$orig_out") \
                | awk -F '\t' -v OFS='\t' -v mn=$mn \
                        'BEGIN {tot=0.0} {diff=sqrt(($1-$2)^2); tot+=diff; print diff,$0}
                         END {printf "\n===Avg diff in %s: %f===\n\n", mn, tot/NR}'
            #TODO1: extract averages and write to a report file
            #TODO2: benchmark speeds
        done
    done <  <(get_sacrebleu_names | head -n $max_tests)
}

main "$@"