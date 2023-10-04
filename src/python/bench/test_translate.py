
import sys
import itertools

from pymarian import Translator

marian = Translator(sys.argv[1])

def batch(s, batch_size=1600):
    it = iter(s)
    while True:
        chunk = list(itertools.islice(it, batch_size))
        if not chunk:
            return
        yield chunk

for b in batch(map(str.strip, sys.stdin)):
    print("\n".join(marian.translate(b)))
