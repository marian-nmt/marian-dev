from pathlib import Path
import os

class Defaults:
    BASE_URL = "https://textmt.blob.core.windows.net/www/marian/metric"

    DEF_CACHE_PATH = Path.home() / '.cache' / 'marian' / 'metric'
    # user might also change this from CLI at runtime
    CACHE_PATH = Path(os.environ['MARIAN_CACHE']) if os.environ.get('MARIAN_CACHE', '').strip() else DEF_CACHE_PATH
    MINI_BATCH = 16
    MAXI_BATCH = 256
    WORKSPACE = 8000
    AVERAGE = 'skip'
    MAX_LENGTH = 512
    FLOAT_PRECISION = 4
    FILE_LOCK_TIMEOUT = 1 * 60 * 60  # seconds => 1 hour
    PROGRESS_BAR = True
    HUGGINGFACE = "huggingface"
    AZURE = "azure"
    COMET_VOCAB_REPO = "microsoft/infoxlm-large"
    # metric   id -> (model_type, huggingface_org/model_id)
    # unbabel agreed to host models within their org and added the same gating/licensing mechanism
    # we hosted bleurt ourself (Apache2.0) on https://huggingface.co/marian-nmt
    KNOWN_METRICS = {
        "bleurt-20": ["bleurt", "marian-nmt/bleurt-20"],
        "wmt20-comet-da": ["comet", "unbabel/wmt20-comet-da-marian"],
        "wmt20-comet-qe-da": ["comet-qe", "unbabel/wmt20-comet-qe-da-marian"],
        "wmt20-comet-qe-da-v2": ["comet-qe", "unbabel/wmt20-comet-qe-da-v2-marian"],
        "wmt21-comet-da": ["comet", "unbabel/wmt21-comet-da-marian"],
        "wmt21-comet-qe-da": ["comet-qe", "unbabel/wmt21-comet-qe-da-marian"],
        "wmt21-comet-qe-mqm": ["comet-qe", "unbabel/wmt21-comet-qe-mqm-marian"],
        "wmt22-comet-da": ["comet", "unbabel/wmt22-comet-da-marian"],
        "wmt22-cometkiwi-da": ["comet-qe", "unbabel/wmt22-cometkiwi-da-marian"],
        "wmt23-cometkiwi-da-xl": ["comet-qe", "unbabel/wmt23-cometkiwi-da-xl-marian"],
        "wmt23-cometkiwi-da-xxl": ["comet-qe", "unbabel/wmt23-cometkiwi-da-xxl-marian"],
    }

    # model type to field order
    MODEL_TYPES = {
        'comet-qe': ('src', 'mt'),
        'bleurt': ('mt', 'ref'),
        'comet': ('src', 'mt', 'ref'),
    }

    DEF_MODEL = 'wmt22-cometkiwi-da'
    DEF_MODEL_TYPE = 'comet-qe'
    DEF_FIELD_ORDER = 'src mt ref'.split()
