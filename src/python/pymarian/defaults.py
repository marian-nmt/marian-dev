from pathlib import Path


class Defaults:
    BASE_URL = "https://textmt.blob.core.windows.net/www/marian/metric"
    CACHE_PATH = Path.home() / '.cache' / 'marian' / 'metric'
    MINI_BATCH = 16
    MAXI_BATCH = 256
    WORKSPACE = 8000
    AVERAGE = 'skip'
    MAX_LENGTH = 512
    FLOAT_PRECISION = 4
    FILE_LOCK_TIMEOUT = 1 * 60 * 60  # seconds => 1 hour
    PROGRESS_BAR = True

    # metric name to model type; lowercase all IDs
    KNOWN_METRICS = {
        "bleurt-20": "bleurt",
        "wmt20-comet-da": "comet",
        "wmt20-comet-qe-da": "comet-qe",
        "wmt20-comet-qe-da-v2": "comet-qe",
        "wmt21-comet-da": "comet",
        "wmt21-comet-qe-da": "comet-qe",
        "wmt21-comet-qe-mqm": "comet-qe",
        "wmt22-comet-da": "comet",
        "wmt22-cometkiwi-da": "comet-qe",
        "xcomet-xl": "comet",
        "xcomet-xxL": "comet",
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
