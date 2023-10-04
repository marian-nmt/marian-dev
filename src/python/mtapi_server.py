#!/usr/bin/env python3

"""
Implements Microsoft's MTAPI (https://docs.microsoft.com/en-us/azure/cognitive-services/translator/quickstart-translator?tabs=python).
"""

import sys
import json
import argparse
from typing import List

from flask import Flask, request
from sacremoses import MosesPunctNormalizer
from sentence_splitter import SentenceSplitter, split_text_into_sentences

import pymarian
SOURCE_LANG = "en"
TARGET_LANG = "de"

# marian = pymarian.Translator(
#     model="model.bin", vocabs=["enu.spm", "enu.spm"],
#     beam_search=2, normalize=1, mini_batch=1, maxi_batch=1,
#     cpu_threads=1, output_approx_knn=[128, 1024]
# ) ---> Translator("--model model.bin --vocabs enu.spm enu.spm --beam-search 2 --normalize 1 --mini-batch 1 --maxi-batch 1 --cpu-threads 1 --output-approx-knn 128 1024")

# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("--marian-flags", "-f")
# parser.add_argument("--doc-model", action="store_true")
# args = parser.parse_args()

norm = MosesPunctNormalizer(lang="en")
splitter = SentenceSplitter(SOURCE_LANG)

class MarianServer:

    def __init__(self, flags=""):
        self.marian = None
        self.flags = flags
        self.reloadMarian()

    def reloadMarian(self):
        if not self.marian:
            self.marian = pymarian.Translator("-m model/model.npz -v model/enu.deu.joint.eos.spm model/enu.deu.joint.eos.spm --mini-batch 1 --maxi-batch 1")

    def translate(self, text: List[str]) -> List[str]:
        self.reloadMarian()

        text = norm.normalize(text)
        inputLines = splitter.split(text)

        if self.marian:
            outputLines = self.marian.translate(inputLines)
            outputText = " ".join(outputLines)

            return outputText

app = Flask(__name__)

server = MarianServer()

@app.route('/translate', methods=["GET", "POST"])
def translate():
    print("GOT A REQUEST", request, str(request), "DATA", request.get_data(), "JSON", request.get_json())

    # if "textType" in request.form and request.form["textType"] == "html":
    #     pass

    request_data = request.get_json()
    outputs = []
    for source in request_data:
        text = source["text"]
        translation = server.translate(text)
        outputs.append(translation)

    response = [{
        "translations": [ { "text": output, "to": TARGET_LANG } for output in outputs ] },
    ]
    print("RESPONSE:", response)

    return json.dumps(response), 200



if __name__ == '__main__':
    app.run()