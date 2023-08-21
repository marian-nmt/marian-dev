#!/bin/bash

parallel --cat -k -j32 --block 10M "sacrebleu <(cut -f 1 {}) < <(cut -f 2 {}) -b -w 4 -sl --format text --metrics bleu"
