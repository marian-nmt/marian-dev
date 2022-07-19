import sys

import pymarian
from sentencepiece import SentencePieceProcessor

sp_src = SentencePieceProcessor(model_file="/home/marcinjd/MTMA/source.spm")
sp_trg = SentencePieceProcessor(model_file="/home/marcinjd/MTMA/target.spm")

marian = pymarian.Translator("/home/marcinjd/MTMA/decoder.yml")

for line in sys.stdin:
    line = " ".join(sp_src.Encode(line, out_type=str))
    line = marian.translate(line)
    print(sp_trg.Decode(line.split(" ")))
