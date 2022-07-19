import sys

from pymarian import Translator
from sentencepiece import SentencePieceProcessor

marian = Translator(sys.argv[1])
sp_src = SentencePieceProcessor(model_file=sys.argv[2])
sp_trg = SentencePieceProcessor(model_file=sys.argv[3])

# oh so ugly
def encode(line, sp):
    return " ".join(sp.Encode(line, out_type=str))

# oh so ugly
def decode(line, sp):
    return sp.Decode(line.split(" "))

for line in sys.stdin:
    input = encode(line, sp_src)
    output = marian.translate(input) # synchronous line-by-line call
    print(decode(output, sp_trg))
