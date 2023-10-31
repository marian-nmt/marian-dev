from pathlib import Path


class Defaults:

    BASE_URL="https://textmt.blob.core.windows.net/www/models/mt-metric"
    CACHE_PATH = Path.home() / '.cache' / 'marian' / 'metrics'
    MINI_BATCH = 16
    MAXI_BATCH = 256
    WORKSPACE = 8000
    AVERAGE = 'skip'
    MAX_LENGTH = 512
    FLOAT_PRECISION = 4

    # NOTE: model names must be lower case for caseless matching
    KNOWN_METRICS = {
        'cometoid22-wmt21': "comet-qe",
        'cometoid22-wmt22': "comet-qe",
        'cometoid22-wmt23': "comet-qe",
        'chrfoid-wmt23': "comet-qe",
        'comet20-da-qe':  "comet-qe",
        'bleurt20': "bleurt",
        'comet20-da': "comet",
        }

    KNOWN_SCHEMA = {
        'comet-qe': 'src+mt',
        'bleurt': 'ref+mt',
        'comet': 'src+mt+ref'
    }

    DEF_MODEL = 'cometoid22-wmt22'
    DEF_SCHEMA = KNOWN_METRICS[DEF_MODEL]