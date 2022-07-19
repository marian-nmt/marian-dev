import sys

from pymarian import Translator
from sentencepiece import SentencePieceProcessor

marian = Translator(sys.argv[1])
sp_src = SentencePieceProcessor(model_file=sys.argv[2])
sp_trg = SentencePieceProcessor(model_file=sys.argv[3])

for line in sys.stdin:
    line = " ".join(sp_src.Encode(line, out_type=str))
    line = marian.translate(line)
    print(sp_trg.Decode(line.split(" ")))
