import sys
import pymarian

marian = pymarian.Translator("/home/marcinjd/MTMA/decoder.yml")

for line in sys.stdin:
    print(marian.translate(line))
