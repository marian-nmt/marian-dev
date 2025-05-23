import logging
from itertools import islice
from pathlib import Path
import sys
from typing import Iterator, List, Optional, Tuple, Union

import _pymarian
import yaml

# this log may be used by submodules, so we declare it here before submodule imports
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

from ._version import __version__
from .defaults import Defaults
from .utils import kwargs_to_cli


class Translator(_pymarian.Translator):
    """Python wrapper for Marian Translator"""

    def __init__(self, cli_string='', **kwargs):
        """Initializes the translator
        :param kwargs: kwargs
        """
        cli_string += ' ' + kwargs_to_cli(**kwargs)
        super().__init__(cli_string.strip())


class Evaluator(_pymarian.Evaluator):
    """Python wrapper for Marian Evaluator"""

    def __init__(self, cli_string='', **kwargs):
        """Initializes the evaluator
        :param kwargs: kwargs
        """
        self._kwargs = kwargs
        self._cli_string = (cli_string + ' ' + kwargs_to_cli(**kwargs)).strip()
        super().__init__(self._cli_string)
        self._config = yaml.safe_load(self.get_model_config())
        log.debug(f'Model config: {self._config}')

    @property
    def model_type(self) -> str:
        return self._config.get('type', None)

    @classmethod
    def new(
        cls,
        model_file: Union[Path, str],
        vocab_file: Union[Path, str] = None,
        devices: Optional[List[int]] = None,
        width=Defaults.FLOAT_PRECISION,
        mini_batch=Defaults.MINI_BATCH,
        maxi_batch=Defaults.MAXI_BATCH,
        like=Defaults.DEF_MODEL_TYPE,
        workspace=Defaults.WORKSPACE,
        max_length=Defaults.MAX_LENGTH,
        cpu_threads=0,
        average: str = Defaults.AVERAGE,
        **kwargs,
    ) -> Iterator[Union[float, Tuple[float, float]]]:
        """A factory function to create an Evaluator with default values.

        :param model_file: path to model file
        :param vocab_file: path to vocabulary file
        :param devices: list of GPU devices to use (optional)
        :param width: number of decimal places to have in output scores
        :param mini_batch: mini-batch size
        :param maxi_batch: maxi-batch size
        :param like: marian metric model like
        :param cpu_threads: number of CPU threads to use
        :param: average: average segment scores to produce system score.
            skip=do not output average (default; segment scores only);
            append=append average at the end;
            only=output the average only (i.e. system score only)
        :return: iterator of scores
        """

        assert Path(model_file).exists(), f'Model file {model_file} does not exist'
        assert Path(vocab_file).exists(), f'Vocab file {vocab_file} does not exist'
        assert like in Defaults.MODEL_TYPES, f'Unknown model type: {like}'
        n_inputs = len(Defaults.MODEL_TYPES[like])
        vocabs = [vocab_file] * n_inputs
        if not kwargs:
            kwargs = {}
        kwargs.update(
            model=model_file,
            vocabs=vocabs,
            devices=devices,
            width=width,
            like=like,
            mini_batch=mini_batch,
            maxi_batch=maxi_batch,
            max_length=max_length,
            max_length_crop=True,
            workspace=workspace,  # negative memory => relative to total memory
            cpu_threads=cpu_threads,
            average=average,
        )
        if kwargs.pop('fp16', False):
            kwargs['fp16'] = ''  # empty string for flag; i.e, "--fp16" and not "--fp16=true"

        # TODO: remove this when c++ bindings supports iterator
        kwargs['average'] = 'skip'
        return cls(**kwargs)

    def evaluate(self, input_lines: Iterator[str], average: str = 'skip', batch_size: Optional[int] = None):
        """Evaluates the input lines and returns the scores

        This function creates mini batches in python and calls the C++ bindings to evaluate the input lines.
        This is a workaround until the C++ bindings support iterator API.

        :param input_lines: iterator of input lines
        :param average: average segment scores to produce system score. Options:
            skip=do not output average (default; segment scores only);
            append=append average at the end;
            only=output the average only (i.e. system score only)
        :param batch_size: batch size (optional; default=2*mini_batch*maxi_batch)
        :return: iterator of scores
        """
        assert average in ('skip', 'append', 'only')
        lines = (line.rstrip('\r\n').split('\t') for line in input_lines)
        if not batch_size:
            mini_batch = self._kwargs.get('mini_batch', Defaults.MINI_BATCH)
            maxi_batch = self._kwargs.get('maxi_batch', Defaults.MAXI_BATCH)
            batch_size = 2 * mini_batch * maxi_batch
            # Sending twice the batch size to avoid starving GPU backend
            # This is a workaround until the C++ bindings support iterator API
        # pymarian bindings does not yet support iterator input, so this function is mini batching here
        def make_maxi_batches(lines, batch_size=batch_size):
            assert batch_size > 0
            while True:
                chunk = list(islice(lines, batch_size))
                if not chunk:
                    return
                yield chunk

        total, count = 0.0, 0
        for batch in make_maxi_batches(lines):
            scores = super().evaluate(batch)
            assert len(scores) == len(batch)
            for score in scores:
                if isinstance(score, (tuple, list)):
                    score = score[0]
                total += score
                count += 1
                if average != 'only':  # skip or append
                    yield score

        if average != 'skip':  # append or only
            yield total / count


class Trainer(_pymarian.Trainer):
    """Python wrapper for Marian Trainer"""

    def __init__(self, cli_string='', **kwargs):
        """Initializes the trainer
        :param kwargs: kwargs
        """
        cli_string += ' ' + kwargs_to_cli(**kwargs)
        super().__init__(cli_string.strip())


class Embedder(_pymarian.Embedder):
    """Python wrapper for Marian Embedder"""

    def __init__(self, cli_string='', **kwargs):
        """Initializes the embedder
        :param kwargs: kwargs
        """
        cli_string += ' ' + kwargs_to_cli(**kwargs)
        super().__init__(cli_string.stip())


def help(*vargs):
    """print help text"""
    args = []
    args += vargs
    if '--help' not in args and '-h' not in args:
        args.append('--help')
    # note: this will print to stdout
    _pymarian.main(args)
    # do not exit, as this is a library function

