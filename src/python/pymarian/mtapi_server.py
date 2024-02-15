#!/usr/bin/env python3

"""
Implements Microsoft's MTAPI (https://docs.microsoft.com/en-us/azure/cognitive-services/translator/quickstart-translator?tabs=python).
"""

import argparse
import json
import logging as log
from typing import List

import pymarian
from flask import Flask, request
from sacremoses import MosesPunctNormalizer
from sentence_splitter import SentenceSplitter

log.basicConfig(level=log.INFO)


class MarianService:
    def __init__(self, source_lang: str, target_lang: str, cli_string: str = None):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.cli_string = cli_string
        self._translator = None  # lazy init

        self.norm = MosesPunctNormalizer(lang="en")
        self.splitter = SentenceSplitter(source_lang)

    @property
    def translator(self):
        if self._translator is None:
            # lazy init
            self._translator = pymarian.Translator(self.cli_string)
        return self._translator

    def translate(self, text: List[str]) -> List[str]:
        """Translates a list of sentences from source to target language."""
        text = self.norm.normalize(text)
        input_lines = self.splitter.split(text)
        output_lines = self.translator.translate(input_lines)
        return " ".join(output_lines)


def attach_routes(app: Flask, service: MarianService):
    @app.route('/translate', methods=["GET", "POST"])
    def translate():
        request_data = request.get_json()
        outputs = []
        for source in request_data:
            text = source["text"]
            translation = service.translate(text)
            outputs.append(translation)
        response = [
            {"translations": [{"text": output, "to": service.target_lang} for output in outputs]},
        ]
        return json.dumps(response), 200


def parse_args():
    SOURCE_LANG = "en"
    TARGET_LANG = "de"

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source-lang", "-s", type=str, default=SOURCE_LANG)
    parser.add_argument("--target-lang", "-t", type=str, default=TARGET_LANG)
    parser.add_argument('args', type=str, help="CLI string for loading marian model")
    parser.add_argument("--port", "-p", type=int, default=5000)
    return vars(parser.parse_args())


def main():
    app = Flask(__name__)
    args = parse_args()
    service = MarianService(
        source_lang=args["source_lang"], target_lang=args["target_lang"], cli_string=args["args"]
    )
    attach_routes(app, service)
    app.run(port=args["port"])


if __name__ == '__main__':
    main()
