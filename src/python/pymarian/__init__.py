from ._version import __version__
import _pymarian

from .utils import kwargs_to_cli

class Translator(_pymarian.Translator):
    """Python wrapper for Marian Translator
    """
    def __init__(self, cli_string='', **kwargs):
        """Initializes the translator
        :param kwargs: kwargs
        """
        cli_string += ' ' + kwargs_to_cli(**kwargs)
        super().__init__(cli_string.strip())


class Evaluator(_pymarian.Evaluator):
    """Python wrapper for Marian Evaluator
    """
    def __init__(self, cli_string='', **kwargs):
        """Initializes the evaluator
        :param kwargs: kwargs
        """
        cli_string += ' ' + kwargs_to_cli(**kwargs)
        super().__init__(cli_string.strip())

class Trainer(_pymarian.Trainer):
    """Python wrapper for Marian Trainer
    """
    def __init__(self, cli_string='', **kwargs):
        """Initializes the trainer
        :param kwargs: kwargs
        """
        cli_string += ' ' + kwargs_to_cli(**kwargs)
        super().__init__(cli_string.strip())

class Embedder(_pymarian.Embedder):
    """Python wrapper for Marian Embedder
    """
    pass

    def __init__(self, cli_string='', **kwargs):
        """Initializes the embedder
        :param kwargs: kwargs
        """
        cli_string += ' ' + kwargs_to_cli(**kwargs)
        super().__init__(cli_string.stip())