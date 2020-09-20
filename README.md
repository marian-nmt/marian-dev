TensorCore 8bit integer decoding for marian.

How to: Compile with:
```bash
cmake .. -DUSE_FBGEMM=ON -DUSE_SENTENCEPIECE=ON -DCOMPILE_CUDA_SM80=OFF -DCOMPILE_CUDA_SM70=OFF -DCOMPILE_CUDA_SM60=OFF -DCOMPILE_CUDA_SM50=OFF -DCOMPILE_CUDA_SM35=OFF
```
Then in order to get better performance than floats, you need to first produce a model with pretrained quantization multipliers and then decode with it on the tensorCores. A script that does all of this looks like:
``` bash
#!/bin/bash

set -e

MARIAN=~/marian-dev-8bitgpu/build_VSCODE

SRC=en
TRG=de
# Preparation part
mkdir -p speed_intgemm
test -e model.npz.best-bleu-detok.alphas.npz || $MARIAN/marian-decoder $@ \
            --relative-paths -m model.npz.best-bleu-detok.npz -v vocab.spm vocab.spm --dump-quantmult \
            -i speed_intgemm/wmt16.$SRC -o speed_intgemm/cpu.wmt16.$TRG \
            --beam-size 1 --mini-batch 32 --maxi-batch 100 --maxi-batch-sort src -w 128 \
            --skip-cost --shortlist lex.s2t.gz 50 50 --cpu-threads 1 \
            --quiet --quiet-translation --log speed_intgemm/cpu.wmt16.log 2> quantmults

test -e model.npz.best-bleu-detok.alphas.npz || $MARIAN/../scripts/alphas/extract_stats.py quantmults model.npz.best-bleu-detok.npz model.npz.best-bleu-detok.alphas.npz
# TensorCore decoding part
for i in 16 17 18 19; do
        /home/s1031254/.local/bin/sacrebleu -t wmt$i -l $SRC-$TRG --echo src > speed_intgemm/wmt$i.$SRC

        $MARIAN/marian-decoder $@ \
            --relative-paths -m model.npz.best-bleu-detok.alphas.npz -v vocab.spm vocab.spm \
            -i speed_intgemm/wmt$i.$SRC -o speed_intgemm/cpu.wmt$i.$TRG \
            --beam-size 1 --mini-batch 32 --maxi-batch 100 --maxi-batch-sort src -w 128 \
            --skip-cost --shortlist lex.s2t.gz 50 50 -d 1 \
            --quiet --quiet-translation --log speed_intgemm/cpu.wmt$i.log --gemm-precision int8tensorAlphaFused

        tail -n1 speed_intgemm/cpu.wmt$i.log
        /home/s1031254/.local/bin/sacrebleu -t wmt$i -l $SRC-$TRG < speed_intgemm/cpu.wmt$i.$TRG | tee speed_intgemm/cpu.wmt$i.$TRG.bleu

done
```
This is the first working version, that delivers about 10% higher performance than fp16 mode for tiny student preset.
TODOs:
 - Tune the GEMM for the different shapes. We use CUTLASS templates that do fused dequantization + bias addition. They have about 10 tunable hyperparameters, it'd be good to fine tune them to the matrix sizes.
 - At the moment compilation only works for one GPU target. Figure out how to make a fat cutlass binary targetting multiple GPU targets at the same time.
 - Make int8 GEMM work together with FP16 mode.
 - Probably lots of other clean ups


Marian
======

[![Build Status CUDA 9](https://img.shields.io/jenkins/s/http/vali.inf.ed.ac.uk/jenkins/view/marian/job/marian-dev-cuda-9.2.svg?label=CUDA%209)](http://vali.inf.ed.ac.uk/jenkins/job/marian-dev-cuda-9.2/)
[![Build Status CUDA 10](https://img.shields.io/jenkins/s/http/vali.inf.ed.ac.uk/jenkins/view/marian/job/marian-dev-cuda-10.1.svg?label=CUDA%2010)](http://vali.inf.ed.ac.uk/jenkins/job/marian-dev-cuda-10.1/)
[![Build Status CPU](https://img.shields.io/jenkins/s/http/vali.inf.ed.ac.uk/jenkins/view/marian/job/marian-dev-cpu.svg?label=CPU)](http://vali.inf.ed.ac.uk/jenkins/job/marian-dev-cpu/)
[![Tests Status](https://img.shields.io/jenkins/s/http/vali.inf.ed.ac.uk/jenkins/view/marian/job/marian-regression-tests.svg?label=tests)](http://vali.inf.ed.ac.uk/jenkins/job/marian-regression-tests/)
[![Latest release](https://img.shields.io/github/release/marian-nmt/marian.svg?label=release)](https://github.com/marian-nmt/marian/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE.md)
[![Twitter](https://img.shields.io/twitter/follow/marian_nmt.svg?style=social)](https://twitter.com/intent/follow?screen_name=marian_nmt)

*Marian* is an efficient Neural Machine Translation framework written in pure
C++ with minimal dependencies.

Named in honour of Marian Rejewski, a Polish mathematician and cryptologist.

Main features:

- Efficient pure C++ implementation
- Fast multi-GPU training and GPU/CPU translation
- State-of-the-art NMT architectures: deep RNN and transformer
- Permissive open source license (MIT)
- [more detail...](https://marian-nmt.github.io/features)

If you use this, please cite:

Marcin Junczys-Dowmunt, Roman Grundkiewicz, Tomasz Dwojak, Hieu Hoang, Kenneth
Heafield, Tom Neckermann, Frank Seide, Ulrich Germann, Alham Fikri Aji, Nikolay
Bogoychev, André F. T. Martins, Alexandra Birch (2018). Marian: Fast Neural
Machine Translation in C++ (http://www.aclweb.org/anthology/P18-4020)

    @InProceedings{mariannmt,
        title     = {Marian: Fast Neural Machine Translation in {C++}},
        author    = {Junczys-Dowmunt, Marcin and Grundkiewicz, Roman and
                     Dwojak, Tomasz and Hoang, Hieu and Heafield, Kenneth and
                     Neckermann, Tom and Seide, Frank and Germann, Ulrich and
                     Fikri Aji, Alham and Bogoychev, Nikolay and
                     Martins, Andr\'{e} F. T. and Birch, Alexandra},
        booktitle = {Proceedings of ACL 2018, System Demonstrations},
        pages     = {116--121},
        publisher = {Association for Computational Linguistics},
        year      = {2018},
        month     = {July},
        address   = {Melbourne, Australia},
        url       = {http://www.aclweb.org/anthology/P18-4020}
    }

## Amun

The handwritten decoder for RNN models compatible with Marian and Nematus has
been superseded by the Marian decoder. The code is available in a separate
repository: https://github.com/marian-nmt/amun

## Website

More information on https://marian-nmt.github.io

- [Quick start](https://marian-nmt.github.io/quickstart)
- [Installation and usage documentation](https://marian-nmt.github.io/docs)
- [Usage examples](https://marian-nmt.github.io/examples)

## Acknowledgements

The development of Marian received funding from the European Union's
_Horizon 2020 Research and Innovation Programme_ under grant agreements
688139 ([SUMMA](http://www.summa-project.eu); 2016-2019),
645487 ([Modern MT](http://www.modernmt.eu); 2015-2017),
644333 ([TraMOOC](http://tramooc.eu/); 2015-2017),
644402 ([HiML](http://www.himl.eu/); 2015-2017),
825303 ([Bergamot](https://browser.mt/); 2019-2021),
the Amazon Academic Research Awards program,
the World Intellectual Property Organization,
and is based upon work supported in part by the Office of the Director of
National Intelligence (ODNI), Intelligence Advanced Research Projects Activity
(IARPA), via contract #FA8650-17-C-9117.

This software contains source code provided by NVIDIA Corporation.
