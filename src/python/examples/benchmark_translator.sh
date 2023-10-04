#!/usr/bin/env bash
set -euo pipefail

mydir=$(dirname $0)

tools=(marian spm_encode spm_decode sacrebleu)
for tool in ${tools[@]}; do
  which $tool >&2 || { echo "$tool not found"; exit 1; }
done

model_dir="/home/marcinjd/MTMA"  # TODO: change to your model directory
source_spm="$model_dir/vocab.spm"
target_spm="$model_dir/vocab.spm"
config="$model_dir/decoder.yml"

source_file="tmp.source.txt"
devices="0 1 2 3"

marian_cmdline="--quiet --config $config -b4 --mini-batch 16 --maxi-batch 100 -d $devices"

[[ -s $source_file ]] || {
   sacrebleu -t wmt20 -l en-de --echo src > $source_file
}

function translate_native {
   spm_encode --model $source_spm  < $source_file \
   |  marian decoder $marian_cmdline \
   | spm_decode --model $target_spm > tmp.translations_native.txt
}

function translate_pybind {
   spm_encode --model $source_spm  < $source_file \
   python mydir/test_translate.py "${marian_cmdline}" \
   | spm_decode --model  $target_spm > tmp.translations_pybind.txt
}

function translate_hf_native {
   cat $source_file | python $mydir/test_hf_raw.py > tmp.translations_hf_raw.txt
}

echo -n "Native: "
time translate_native
echo
echo -n "Pybind: "
time translate_pybind
echo
#echo -n "Transformers (native): "
#time translate_hf_native
