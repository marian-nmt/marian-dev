#!/usr/bin/env python3

"""
Implements Microsoft's MTAPI (https://docs.microsoft.com/en-us/azure/cognitive-services/translator/quickstart-translator?tabs=python).
"""

import sys
import pymarian

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from sacremoses import MosesPunctNormalizer
#from sentence_splitter import SentenceSplitter, split_text_into_sentences

# marian = pymarian.Translator(
#     model="model.bin", vocabs=["enu.spm", "enu.spm"],
#     beam_search=2, normalize=1, mini_batch=1, maxi_batch=1,
#     cpu_threads=1, output_approx_knn=[128, 1024]
# ) ---> Translator("--model model.bin --vocabs enu.spm enu.spm --beam-search 2 --normalize 1 --mini-batch 1 --maxi-batch 1 --cpu-threads 1 --output-approx-knn 128 1024")

#norm = MosesPunctNormalizer(lang="en")
#splitter = SentenceSplitter("en")

class MarianServer(ThreadingHTTPServer):
    def __init__(self, server_address, handler):
        self.marian = None
        self.reloadMarian()

        super().__init__(server_address, handler)

    def reloadMarian(self):
        if not self.marian:
            self.marian = pymarian.Translator(command)

    def translate(self, text):
        return text.reverse()

        self.reloadMarian()
        # inputText = norm.normalize(self.input.toPlainText())
        # inputLines = splitter.split(inputText)
        # if self.marian:
        #     outputLines = self.marian.translate(inputLines)
        #     outputText = "\n".join(outputLines)
        #     self.output.setPlainText(outputText)


class RequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        text = "This is a POST"

        print("POST")
        print(self.server.translate(text))

    def do_GET(self):
        print("GET")
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(bytes("<html><head><title>https://pythonbasics.org</title></head>", "utf-8"))
        self.wfile.write(bytes("<p>Request: %s</p>" % self.path, "utf-8"))
        self.wfile.write(bytes("<body>", "utf-8"))
        self.wfile.write(bytes("<p>This is an example web server.</p>", "utf-8"))
        self.wfile.write(bytes("</body></html>", "utf-8"))

def main(args):

    server = MarianServer((args.hostname, args.port), RequestHandler)
    print("Server started http://%s:%s" % (hostName, serverPort))

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass

    server.server_close()
    print("Server stopped.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hostname", default="localhost")
    parser.add_argument("--port", "-p", default=8976)
    args = parser.parse_args()

    main(args)
