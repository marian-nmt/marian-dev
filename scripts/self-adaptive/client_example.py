#!/usr/bin/env python

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
