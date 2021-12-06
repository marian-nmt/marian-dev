#!/usr/bin/env python

# This is an example for using self-adaptive translation in server mode.
#
# To run:
# 1. Start self-adaptive Marian in server mode, e.g.:
#     ./build/marian-adaptive-server -p 8080 -m model.npz -v vocap.yaml vocab.yaml \
#         --after-batches 10 --after-epochs 10 --learn-rate 0.1 --mini-batch 15 # other options
# 2. In a new shell, run this script:
#     python3 ./scripts/self-adaptive/client_exmaple.py -p 8080
#
# For a more extensive example, see https://github.com/marian-cef/marian-examples/tree/master/adaptive

from __future__ import print_function, unicode_literals, division

import sys
import time
import argparse
import json

from websocket import create_connection


def translate(batch, port=8080):
    ws = create_connection("ws://localhost:{}/translate".format(port))
    ws.send(batch)
    result = ws.recv()
    ws.close()
    return result.rstrip()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=8080)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # List of input sentences separated by a new line character
    inputs = "this is an example\nthe second sentence\nno context provided"
    # For each input sentence a list of parallel sentences can be provided as a
    # list of source and target sentences.
    contexts = [
        # Source-side context for the first input sentence
        ["this is a test\nthese are examples",
        # Target-side context for the first input sentence
            "das ist ein test\ndies sind Beispiele"],
        # Only one example is given as a context for the second input sentence
        ["the next sentence",
            "der nächste Satz"],
        # No context for the third input sentence
        []
    ]

    input_data = {'input': inputs, 'context': contexts}
    input_json = json.dumps(input_data)

    output_json = translate(input_json, port=args.port)
    output_data = json.loads(output_json)
    print(output_data['output'])
