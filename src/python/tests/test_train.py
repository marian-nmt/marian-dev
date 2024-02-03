import tarfile
import tempfile
import urllib.request
from pathlib import Path

from pymarian import Trainer
from pymarian.utils import get_known_model

QUIET = False

TMP_DATA_DIR = Path.home() / 'tmp' / 'marian-tests'
DATA_URL = "https://textmt.blob.core.windows.net/www/data/marian-tests-data.tgz"


def setup():
    ok_file = TMP_DATA_DIR / '_OK'
    if not TMP_DATA_DIR.exists() or not ok_file.exists():
        TMP_DATA_DIR.mkdir(parents=True, exist_ok=True)

        print("Downloading data package...")
        with urllib.request.urlopen(DATA_URL) as response:
            with tarfile.open(fileobj=response, mode="r|gz") as tar:
                tar.extractall(path=TMP_DATA_DIR)
        ok_file.touch()
        print("Done.")


setup()


def test_train_comet_qe():
    data_dir = TMP_DATA_DIR / 'marian-tests-data/deu-eng'
    vocab_file = data_dir / 'vocab.8k.spm'
    classe_file = data_dir / 'classes4f.txt'
    train_file = data_dir / 'sample.5k.chrfoid-deu-eng.tsv'
    # pretrained_model, vocab_file = get_known_model("chrfoid-wmt23")
    assert classe_file.exists()
    assert vocab_file.exists()
    assert train_file.exists()

    args = {
        'dim_emb': 128,
        'enc_depth': 3,
        'dec_depth': 1,
        'tied_embeddings_all': True,
        'transformer_heads': 2,
        'transformer_dim_ffn': 256,
        'transformer_ffn_activation': 'relu',
        'transformer_dropout': 0.1,
        'cost_type': 'ce-mean',
        'max_length': 80,
        'mini_batch_fit': False,
        'maxi_batch': 256,
        'optimizer_params': [0.9, 0.98, '1e-09'],
        'sync_sgd': True,
        'learn_rate': 0.0003,
        'lr_decay_inv_sqrt': [16000],
        'lr_warmup': 16000,
        'label_smoothing': 0.1,
        'clip_norm': 0,
        'exponential_smoothing': 0.0001,
        'early_stopping': 2,
        'keep_best': True,
        'beam_size': 2,
        'normalize': 1,
        'valid_metrics': ['perplexity'],
        'valid_mini_batch': 16,
        'mini_batch': 8,
        'after': '400u',
        'valid_freq': '200u',
        'disp_freq': 100,
        'disp_first': 4,
        'save_freq': '200u',
        'quiet': QUIET,
        #'like': 'comet-qe',   # only supported at inference; for training, see task and input_types
        'task': 'comet-qe',
        'input_types': ['class', 'sequence', 'sequence'],  # required for training
        #'pretrained_model': pretrained_model,     # for finetuning; not using it because its too big for tests
        'train_sets': [train_file],  # TSV file having 3 columns: class sequence sequence
        'tsv': True,
        'tsv-fields': 3,  # or it will complain that vocabs and train_sets should be one to one map
        'vocabs': [classe_file, vocab_file, vocab_file],  # class sequence sequence
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        save_at = tmpdir + '/model.npz'
        trainer = Trainer(model=save_at, **args)
        trainer.train()


def test_train_transformer_nmt():
    data_dir = TMP_DATA_DIR / 'marian-tests-data/deu-eng'
    vocab_file = data_dir / 'vocab.8k.spm'
    train_prefix = str(data_dir / 'sample.5k')
    src_lang = "deu"
    tgt_lang = "eng"
    train_src = train_prefix + "." + src_lang
    train_tgt = train_prefix + "." + tgt_lang

    # these are taken from regression-tests repo and simplified
    args = {
        'type': 'transformer',
        'dim_emb': 128,
        'enc_depth': 3,
        'dec_depth': 1,
        'tied_embeddings_all': True,
        'transformer_heads': 2,
        'transformer_dim_ffn': 256,
        'transformer_ffn_activation': 'relu',
        'transformer_dropout': 0.1,
        'cost_type': 'ce-mean-words',
        'max_length': 80,
        'mini_batch_fit': False,
        'maxi_batch': 256,
        'optimizer_params': [0.9, 0.98, '1e-09'],
        'sync_sgd': True,
        'learn_rate': 0.0003,
        'lr_decay_inv_sqrt': [16000],
        'lr_warmup': 16000,
        'label_smoothing': 0.1,
        'clip_norm': 0,
        'exponential_smoothing': 0.0001,
        'early_stopping': 2,
        'keep_best': True,
        'beam_size': 2,
        'normalize': 1,
        'valid_metrics': ['ce-mean-words', 'bleu', 'perplexity'],
        'valid_mini_batch': 16,
        'mini_batch': 8,
        'after': '400u',  # stop after 500 updates
        'valid_freq': '200u',  # validate every 250 updates
        'disp_freq': 100,
        'disp_first': 4,
        'save_freq': '200u',
        'vocabs': [vocab_file, vocab_file],
        'train_sets': [train_src, train_tgt],
        'quiet': QUIET,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        save_at = tmpdir + '/model.npz'
        trainer = Trainer(model=save_at, **args)
        trainer.train()
