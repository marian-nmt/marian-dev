from pathlib import Path

from pymarian import Translator

from . import BASE_ARGS


def test_ende():
    # TODO: download model from blob storage
    model_dir = Path.home() / 'tmp/marian-eng-deu'
    model_file = str(model_dir / 'model.bin')
    vocab_file = str(model_dir / 'vocab.spm')
    args = BASE_ARGS | dict(models=model_file, vocabs=[vocab_file, vocab_file])
    translator = Translator(**args)
    hyp = translator.translate("Hello. Good morning.")
    assert hyp == "Hallo. Guten Morgen."
