#!/usr/bin/env sh

function translate_native {
 sacrebleu -t wmt20 -l en-de --echo src | \
   spm_encode --model /home/marcinjd/MTMA/source.spm |  ../build/temp.linux-x86_64-3.8/PyMarian/marian-decoder --quiet \
     -c  /home/marcinjd/MTMA/decoder.yml -b4 --mini-batch 16 --maxi-batch 100 -d 0 1 2 3 \
   | spm_decode --model  /home/marcinjd/MTMA/target.spm > translations_native.txt
}

function translate_python {
   sacrebleu -t wmt20 -l en-de --echo src | \
   spm_encode --model /home/marcinjd/MTMA/source.spm | \
   python test_translate.py '--config /home/marcinjd/MTMA/decoder.yml -b4 --mini-batch 16 --quiet --maxi-batch 100 -d 0 1 2 3' \
   | spm_decode --model  /home/marcinjd/MTMA/target.spm > translations_python.txt
}

echo -n "Native: "
time translate_native
echo
echo -n "Pybind: "
time translate_python