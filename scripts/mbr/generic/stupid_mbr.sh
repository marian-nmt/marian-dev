#!/bin/bash

if [[ "$1" = "--help" ]]
then
cat >&2 <<END
This script can be used to do "stupid" MBR (i.e. all-vs-all MBR with any reference-based metric specfied in the metrics folder).
The subscipt in the metrics folder need to be able to calculate sentence-level results. This should be done as efficiently as possible
in order to score all NxN variants (where N is sample size). The explode_collape.pl script below does some smart deduping as far as 
possible, but the complexity will still be close to NxN.

Example usage:

# prepare the sample
sacrebleu -t wmt21 -l en-de --echo src | perl -pe '\$_ = \$_ x 128' > wmt21.128.src
cat wmt21.128.src | ~/marian-dev/build/marian-decoder -m translation-model.npz \
 -v translation-model-vocab.{spm,spm} -b1 --mini-batch 32 --maxi-batch 100 --maxi-batch-sort src \
 --max-length 256 --max-length-crop -d all --output-sampling > wmt21.128.out

# run MBR, here with ChrF
cat wmt21.128.out | ~/marian-dev/scripts/mbr/generic/stupid_mbr.sh 128 128 chrf > wmt21.128.sorted.out

# select the top translation according to ChrF MBR and evaluate result
cat wmt21.128.sorted.out | grep ^BEST | cut -f 3 | sacrebleu -t wmt21 -l en-de --metrics bleu chrf -w 2 --format text

END
exit
fi

num_samples=${1:-128}
num_references=${2:-$num_samples}
metric=${3:-bleu}
gpus=${4:-8}

scriptPath=$(dirname $0)
tmp=$(mktemp -d)

cat \
| tee >(wc -l > $tmp/lines_input) \
| pigz > $tmp/input.txt.gz

lines_input=$(cat $tmp/lines_input)

>&2 echo "Computing $metric scores"

pigz -dc $tmp/input.txt.gz \
| pv -ptel -s $lines_input \
| perl $scriptPath/explode_collapse.pl $num_samples $num_references 2>/dev/null \
| tee >(cut -f 1,2,3 > $tmp/ids) \
| cut -f 4,5 \
| $scriptPath/metrics/$metric.sh $gpus \
> $tmp/scores

>&2 echo "Computing MBR scores"

pigz -dc $tmp/input.txt.gz \
| pv -ptel -s $lines_input \
| perl $scriptPath/rescore.pl $num_samples $num_references $tmp/ids $tmp/scores

rm -rf $tmp
>&2 echo "Done"
