#!/usr/bin/env python3

import sys
import itertools

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from parallelformers import parallelize

def translate(model, tokenizer, batch):
    inputs = tokenizer(batch, return_tensors="pt", padding=True)
    outputs = model.generate(**inputs, num_beams=4)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def batch(s, batch_size=1600):
    it = iter(s)
    while True:
        chunk = list(itertools.islice(it, batch_size))
        if not chunk:
            return
        yield chunk


if __name__ == "__main__":
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    parallelize(model, num_gpus=4, fp16=True)

    for b in batch(map(str.strip, sys.stdin), 128):
        print("\n".join(translate(model, tokenizer, b)))
