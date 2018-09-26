#!/usr/bin/env python

from __future__ import print_function, unicode_literals, division

import sys
import time
import argparse
import json

from websocket import create_connection


def translate(batch, port=8080):
    print(port)
    print(batch.rstrip())
    ws = create_connection("ws://localhost:{}/translate".format(port))
    ws.send(batch)
    result = ws.recv()
    print(result.rstrip())
    ws.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=8080)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    inputs = "this is an example"
    contexts = [
        ["this is a test\nthese are examples",
            "das ist ein test\ndies sind Beispiele"]
    ]

    json_data = {'input': inputs, 'context': contexts}
    json_text = json.dumps(json_data)
    translate(json_text, port=args.port)

    # count = 0
    # batch = ""
    # for line in sys.stdin:
    # count += 1
    # if count == args.batch_size:
    # translate(batch, port=args.port)
    # count = 0
    # batch = ""

    # if count:
    # translate(batch, port=args.port)
