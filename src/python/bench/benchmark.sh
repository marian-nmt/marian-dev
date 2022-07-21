#!/usr/bin/env sh

function translate_native {
 sacrebleu -t wmt20 -l en-de --echo src | \
   spm_encode --model /home/marcinjd/MTMA/source.spm |  ../_skbuild/linux-x86_64-3.8/cmake-build/marian-decoder --quiet \
     -c  /home/marcinjd/MTMA/decoder.yml -b4 --mini-batch 16 --maxi-batch 100 -d 0 1 2 3 \
   | spm_decode --model  /home/marcinjd/MTMA/target.spm > translations_native.txt
}

function translate_pybind {
   sacrebleu -t wmt20 -l en-de --echo src | \
   spm_encode --model /home/marcinjd/MTMA/source.spm | \
   python test_translate.py '--config /home/marcinjd/MTMA/decoder.yml -b4 --mini-batch 16 --quiet --maxi-batch 100 -d 0 1 2 3' \
   | spm_decode --model  /home/marcinjd/MTMA/target.spm > translations_pybind.txt
}

function translate_hf_native {
   sacrebleu -t wmt20 -l en-de --echo src | python test_hf_raw.py > translations_hf_raw.txt
}

echo -n "Native: "
time translate_native
echo
echo -n "Pybind: "
time translate_pybind
echo
echo -n "Transformers (native): "
time translate_hf_native
