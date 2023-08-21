#!/bin/bash

OPTIONS=$(getopt -o t:s:o:n:m:g:h --long hyps:,source:,output:,num_hyps:,model:,num_gpus:,help -- "$@")
eval set -- "$OPTIONS"

while true; do
  case "$1" in
    -t|--hyps)
      >&2 echo "Option hyps=$2"
      hyps_file=$2
      shift 2;;
    -s|--source)
      >&2 echo "Option source=$2"
      source_file=$2
      shift 2;;
    -o|--output)
      >&2 echo "Option output=$2"
      out_file=$2
      shift 2;;
    -n|--num_hyps)
      >&2 echo "Option num_hyps=$2"
      num_hyps=$2
      shift 2;;
    -m|--model)
      >&2 echo "Option model=$2"
      comet_model=$2
      shift 2;;
    -g|--num_gpus)
      >&2 echo "Option num_gpus=$2"
      num_gpus=$2
      shift 2;;
    -h|--help)
      help=1
      shift;;
    --)
      shift; break;;
    *)
      >&2 echo "Internal error!" ; exit 1 ;;
  esac
done

if [[ "$help" = "1" ]]
then
cat >&2 <<END
This script is for efficient MBR with COMET. COMET allows to embed source and hypotheses separatly which makes it very easy to optimize.
Only the final embbedings are used to create the NxN scores.

Example usage:

# prepare the source and samples
sacrebleu -t wmt21 -l en-de --echo src > wmt21.src
cat wmt21.src | perl -pe '\$_ = \$_ x 128' > wmt21.128.src
cat wmt21.128.src | ~/marian-dev/build/marian-decoder -m translation-model.npz \
 -v translation-model-vocab.{spm,spm} -b1 --mini-batch 32 --maxi-batch 100 --maxi-batch-sort src \
 --max-length 256 --max-length-crop -d all --output-sampling > wmt21.128.out

# run MBR with COMET
cat wmt21.128.out | ~/marian-dev/scripts/mbr/comet/comet_mbr.sh -m wmt20-comet-da.npz -n 128 -s wmt21.src -g 8 > wmt21.128.mbr.out
cat wmt21.128.mbr.out | cut -f 4 | sacrebleu -t wmt21 -l en-de --metrics bleu chrf -w 2 --format text

END
exit
fi


hyps_file=${hyps_file:-/dev/stdin}
out_file=${out_file:-/dev/stdout}
num_hyps=${num_hyps:-128}
comet_model=${comet_model:-wmt20-comet-da.npz}
num_gpus=${num_gpus:-8}

script_path=$(dirname $0)
marian=$script_path/../../../build/marian

comet_path=$(dirname $comet_model)
devices=$(seq 0 $(($num_gpus-1)))

tmp=/tmp

# create temporary files and delete them right after, use file descriptor instead
# (will free disk space after script ends, even when interrupted)
samples=$(mktemp $tmp/samples.XXXXXX)
exec 3>"$samples"
rm "$samples"
samples=/dev/fd/3

source=$(mktemp $tmp/source.XXXXXX)
exec 4>"$source"
rm "$source"
source=/dev/fd/4

source_embeddings=$(mktemp $tmp/source.embeddings.bin.XXXXXX)
exec 5>"$source_embeddings"
rm "$source_embeddings"
source_embeddings=/dev/fd/5

hyps_embeddings=$(mktemp $tmp/sample.embeddings.bin.XXXXXX)
exec 6>"$hyps_embeddings"
rm "$hyps_embeddings"
hyps_embeddings=/dev/fd/6

# done with creating temporary files

lines_hyps=$(cat $hyps_file | tee $samples | wc -l)
lines_source=$(cat $source_file | tee $source | wc -l)

>&2 echo "Computing source embeddings ($lines_source lines) with $comet_model"

cat $source \
| pv -ptel -s $lines_source \
| $marian embed -m $comet_model -v $comet_path/roberta-vocab.spm \
  --like roberta -d $devices --fp16 --binary --quiet \
> $source_embeddings

>&2 echo "Computing sample embeddings ($lines_hyps lines, $num_hyps per sentence) with $comet_model"

cat $samples \
| pv -ptel -s $lines_hyps \
| $marian embed -m $comet_model -v $comet_path/roberta-vocab.spm \
  --like roberta -d $devices --fp16 --binary --quiet \
> $hyps_embeddings
 
>&2 echo "Computing MBR scores"

cat $samples \
| pv -ptel -s $lines_hyps \
| python $script_path/comet_mbr_with_embeddings.py \
  -m $comet_model -s $source_embeddings -t $hyps_embeddings \
  --num_source $lines_source --num_hyps $num_hyps \
  -d $devices --batch_size 128 --fp16 \
> $out_file

>&2 echo "Done"
