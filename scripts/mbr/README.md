# Some notes on MBR

All of this is experimental, use at your own risk.

## MBR with COMET

This concerns the scipts in the `comet` folder:

This script is for efficient MBR with COMET. COMET allows to embed source and hypotheses separatly which makes it very easy to optimize.
Only the final embbedings are used to create the NxN scores.

Example usage:

### prepare the source and samples
sacrebleu -t wmt21 -l en-de --echo src > wmt21.src
cat wmt21.src | perl -pe '\$_ = \$_ x 128' > wmt21.128.src
cat wmt21.128.src | ~/marian-dev/build/marian-decoder -m translation-model.npz \
 -v translation-model-vocab.{spm,spm} -b1 --mini-batch 32 --maxi-batch 100 --maxi-batch-sort src \
 --max-length 256 --max-length-crop -d all --output-sampling > wmt21.128.out

### run MBR with COMET
cat wmt21.128.out | ~/marian-dev/scripts/mbr/comet/comet_mbr.sh -m wmt20-comet-da.npz -n 128 -s wmt21.src -g 8 > wmt21.128.mbr.out
cat wmt21.128.mbr.out | cut -f 4 | sacrebleu -t wmt21 -l en-de --metrics bleu chrf -w 2 --format text


## "Stupid" MBR (generic)

This concerns the scipts in the `generic` folder

This script can be used to do "stupid" MBR (i.e. all-vs-all MBR with any reference-based metric specfied in the metrics folder).
The subscipt in the metrics folder need to be able to calculate sentence-level results. This should be done as efficiently as possible
in order to score all NxN variants (where N is sample size). The explode_collape.pl script below does some smart deduping as far as 
possible, but the complexity will still be close to NxN.

Example usage:

### prepare the sample
```
sacrebleu -t wmt21 -l en-de --echo src | perl -pe '\$_ = \$_ x 128' > wmt21.128.src
cat wmt21.128.src | ~/marian-dev/build/marian-decoder -m translation-model.npz \
 -v translation-model-vocab.{spm,spm} -b1 --mini-batch 32 --maxi-batch 100 --maxi-batch-sort src \
 --max-length 256 --max-length-crop -d all --output-sampling > wmt21.128.out
```

### run MBR, here with ChrF
```
cat wmt21.128.out | ~/marian-dev/scripts/mbr/generic/stupid_mbr.sh 128 128 chrf > wmt21.128.sorted.out
```

### select the top translation according to ChrF MBR and evaluate result

```
cat wmt21.128.sorted.out | grep ^BEST | cut -f 3 | sacrebleu -t wmt21 -l en-de --metrics bleu chrf -w 2 --format text
```