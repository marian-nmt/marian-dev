import tarfile
import urllib.request
from pathlib import Path

from pymarian import Translator

from . import BASE_ARGS

DATA_URL = "http://data.statmt.org/romang/marian-regression-tests/models/wngt19.tar.gz"
DATA_DIR = Path(__file__).parent.parent / "data" / "wngt19"


def setup():
    flag_file = DATA_DIR / ".downloaded"
    if flag_file.exists():
        print("Data already downloaded. Setup skipped...")
        return
    print(f"Downloading {DATA_URL} to {DATA_DIR}")
    request = urllib.request.urlopen(DATA_URL)
    with tarfile.open(fileobj=request, mode="r|gz") as tar:
        tar.extractall(path=DATA_DIR.parent)
    flag_file.touch()


setup()


def test_ende():

    model_file = str(DATA_DIR / 'model.base.npz')
    vocab_file = str(DATA_DIR / 'en-de.spm')
    args = BASE_ARGS | dict(models=model_file, vocabs=[vocab_file, vocab_file])
    translator = Translator(**args)
    hyp = translator.translate("Hello. Good morning.")
    assert hyp == "Hallo , Guten Morgen ."


def test_ende_force_decode():

    model_file = str(DATA_DIR / 'model.base.npz')
    vocab_file = str(DATA_DIR / 'en-de.spm')
    args = BASE_ARGS | dict(models=model_file, vocabs=[vocab_file, vocab_file], quiet=True)
    translator = Translator(**args)
    hyp = translator.translate("Hello. Good morning.")
    assert hyp == "Hallo , Guten Morgen ."

    force_decode_config = dict(force_decode=True, tsv=True, tsv_fields=2)
    hyp = translator.translate("Hello. Good morning.\tIsch", **force_decode_config)
    assert hyp == "Isch am Guten Morgen ."
