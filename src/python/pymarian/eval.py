#!/usr/bin/env python
#
# This is a python wrapper for marian evaluate command
#
import argparse
import logging as log
import sys
from itertools import zip_longest
from pathlib import Path
from typing import Iterator, List

from . import Evaluator, __version__
from .defaults import Defaults
from .utils import get_model_path, get_vocab_path

log.basicConfig(level=log.INFO)
DEBUG_MODE = False


def parse_args():
    parser = argparse.ArgumentParser(
        "pymarian-eval",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog='More info at https://github.com/marian-nmt/marian-dev. '
        f'This CLI is loaded from {__file__} (version: {__version__})',
    )

    known_metrics = ', '.join(Defaults.KNOWN_METRICS.keys())
    parser.add_argument(
        '-m',
        '--model',
        help=f'Model name, or path. Known models: {known_metrics}',
        default=Defaults.DEF_MODEL,
        type=str,
    )
    parser.add_argument('-v', '--vocab', help=f'Vocabulary file', type=Path)
    parser.add_argument(
        '-l',
        '--like',
        help='Model type. Required if --model is a local file (auto inferred for known models)',
        type=str,
        choices=list(Defaults.MODEL_TYPES.keys()),
    )
    parser.add_argument('-V', '--version', action="version", version=f"%(prog)s {__version__}")

    parser.add_argument(
        '-',
        '--stdin',
        action='store_true',
        help='Read input from stdin. TSV file with following format: \
                        QE metrics: "src<tab>mt", Ref based metrics ref: "src<tab>mt<tab>ref" or "mt<tab>ref"',
    )
    parser.add_argument('-t', '--mt', dest='mt_file', help='MT output file. Ignored when --stdin', type=Path)
    parser.add_argument('-s', '--src', dest='src_file', help='Source file. Ignored when --stdin', type=Path)
    parser.add_argument('-r', '--ref', dest='ref_file', help='Ref file. Ignored when --stdin', type=Path)
    parser.add_argument(
        '-f',
        '--fields',
        dest='user_fields',
        metavar='FIELD',
        nargs='+',
        choices=['src', 'mt', 'ref'],
        help='Input fields, an ordered sequence of {src, mt, ref}',
        default=Defaults.DEF_FIELD_ORDER,
        type=str,
    )
    parser.add_argument('-o', '--out', default=sys.stdout, help='output file', type=argparse.FileType('w'))
    parser.add_argument(
        '-a',
        '--average',
        choices=('skip', 'append', 'only'),
        default='skip',
        help='Average segment scores to produce system score.'
        ' skip=do not output average (default; segment scores only);'
        ' append=append average at the end; '
        ' only=output the average only (i.e. system score only)',
    )

    parser.add_argument('-w', '--width', default=4, help='Output score width', type=int)
    parser.add_argument('--debug', help='Debug or verbose mode', action='store_true')
    parser.add_argument('--fp16', help='Enable FP16 mode', action='store_true')
    parser.add_argument('--mini-batch', default=16, help='Mini-batch size', type=int)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-d', '--devices', nargs='*', type=int, help='GPU device IDs')
    group.add_argument(
        '-c', '--cpu-threads', default=None, type=int, help='Use CPU threads. 0=use GPU device 0'
    )
    parser.add_argument('-ws', '--workspace', default=8000, help='Workspace memory', type=int)
    parser.add_argument(
        '-pc', '--print-cmd', action="store_true", help="Print marian evaluate command and exit"
    )
    parser.add_argument('--cache', help='Cache directory for storing models', type=Path, default=Defaults.CACHE_PATH)

    args = parser.parse_args()
    return vars(args)


def find_field_ordering(expected_fields: List[str], given_fields: List[str]) -> List[int]:
    """Find the order of fields in given_fields to match expected_fields

    :param expected_fields: list of expected fields
    :param given_fields: list of given fields
    :return: list of indices to select from given_fields to match expected_fields
    :throws ValueError: if any expected field is missing in given_fields
    """

    missing_fields = set(expected_fields) - set(given_fields)
    if missing_fields:
        raise ValueError(
            f'Required fields are missing: {missing_fields} [expected: {expected_fields}, given: {given_fields}]'
        )
    field_order = []
    for name in expected_fields:
        idx = given_fields.index(name)
        assert idx >= 0, f'Field {name} not found in {given_fields}. Please check --fields argument'  # this should never happen
        field_order.append(idx)
    return field_order


def reorder_fields(lines: Iterator[str], field_order: List[int]) -> Iterator[str]:
    """Reorder fields in each line according to field_order

    :param lines: input lines
    :param field_order: list of indices to reorder fields
    :return: lines with fields reordered
    :throws ValueError: if any line has missing fields
    """
    max_column = max(field_order)
    for line_num, line in enumerate(lines, start=1):
        fields = line.rstrip('\r\n').split('\t')
        if len(fields) <= max_column:
            raise ValueError(
                f'Expected at least {max_column} columns, but got {len(fields)} in line {line_num}'
            )
        yield '\t'.join(fields[i] for i in field_order)


def read_input(
    stdin=False,
    src_file=None,
    mt_file=None,
    ref_file=None,
    expected_fields=Defaults.DEF_FIELD_ORDER,
    user_fields=Defaults.DEF_FIELD_ORDER,
):
    """Read input files and reorder fields if necessary.

    This function modifies args dictionary in place.
    :param args: command line arguments
    :param model_id: model ID
    :param schema: schema to use for the model
    """

    n_inputs = len(expected_fields)
    assert 1 <= n_inputs <= 3, f'Invalid : {expected_fields}'

    if stdin:
        assert 1 <= len(user_fields) <= 3
        reorder_idx = find_field_ordering(expected_fields, user_fields)
        log.info(f'Input field mappings: {reorder_idx}; expected: {expected_fields}, given: {user_fields}')
        return reorder_fields(sys.stdin, reorder_idx)

    n_inputs = len(expected_fields)
    assert mt_file.exists(), 'File with hypotheses {mt_file} does not exist'
    if 'src' in expected_fields:
        assert src_file, f'Source file is required'
        assert src_file.exists(), f'{src_file} does not exist'
    if 'ref' in expected_fields:
        assert ref_file, f'Reference file is required'
        assert ref_file.exists(), f'{ref_file} does not exist'

    if expected_fields == ('src', 'mt'):
        input_lines = zip_longest(open(src_file), open(mt_file))
    elif expected_fields == ('mt', 'ref'):
        input_lines = zip_longest(open(mt_file), open(ref_file))
    elif expected_fields == ('src', 'mt', 'ref'):
        input_lines = zip_longest(open(src_file), open(mt_file), open(ref_file))
    else:
        raise ValueError(f'Unknown schema {expected_fields}')

    def _validate_and_join():
        for row in input_lines:
            assert len(row) == n_inputs, f'Expected {n_inputs} columns, but got {len(row)}'
            for col in row:
                assert col is not None, f'Expected {n_inputs} columns, but got {len(row)}'
            line = '\t'.join(col.strip() for col in row)
            yield line

    return _validate_and_join()


def main(**args):
    args = args or parse_args()
    if args.pop('debug'):
        log.getLogger().setLevel(log.DEBUG)
        global DEBUG_MODE
        DEBUG_MODE = True
        log.debug(args)
    else:
        args['quiet'] = ''
    Defaults.CACHE_PATH = args.pop('cache')

    model_id = args.pop('model')
    model_path = Path(model_id)
    vocab_path = args.pop('vocab')
    if vocab_path:  # if user gave this arg, it must be a valid arg
        assert vocab_path.exists(), f'Vocabulary file {vocab_path} does not exist'

    # if model arg is local path
    if model_path.suffix.lower() in ('.npz', '.bin'):
        assert model_path.exists() and model_path.is_file(), f'Model file {model_path} does not exist'
        model_id = model_path.stem
        assert args.get('like'), f'--like is required when --model is a local file'
        if not vocab_path:  # if vocab is not given, resolve it from model directory
            vocab_path = model_path.parent / 'vocab.spm'
            if not vocab_path.exists():
                raise Exception(
                    f'Vocabulary file {vocab_path} does not exist. Plese sepcify it with --vocab option.'
                )
    else:  # assume it is ID and resolve path from cache
        model_id = model_id.lower()
        try:
            model_path = get_model_path(model_id)
            if not vocab_path:  # if vocab is not given, resolve it from cache
                vocab_path = get_vocab_path(model_id)
            args['like'] = Defaults.KNOWN_METRICS.get(model_id, [Defaults.DEF_MODEL_TYPE])[0]
        except ValueError as e:
            raise ValueError(f'Invalid model ID: {model_id}') from e

    args['model_file'] = model_path
    args['vocab_file'] = vocab_path

    out = args.pop('out')
    width = args.pop('width', Defaults.FLOAT_PRECISION)
    average = args.pop('average', Defaults.AVERAGE)
    print_cmd = args.pop('print_cmd', False)

    input_args = ('stdin', 'src_file', 'mt_file', 'ref_file', 'user_fields')
    input_args = {k: args.pop(k) for k in input_args}
    input_args['expected_fields'] = Defaults.MODEL_TYPES[args['like']]
    model_args = args

    evaluator = Evaluator.new(**model_args)
    if evaluator.model_type != args['like']:
        log.warning(f'Config model type is {evaluator.model_type}, but given: {args["like"]}')

    input_lines = read_input(**input_args)
    cmd_line = "marian evaluate " + evaluator._cli_string
    if print_cmd:  # print the command and exit
        print(cmd_line)
        return
    else:
        log.info("CLI:\t" + cmd_line)

    scores = evaluator.evaluate(input_lines, average=average)

    for i, score in enumerate(scores, start=1):
        if isinstance(score, (tuple, list)):
            score = score[0]  # the first score
        out.write(f'{score:.{width}f}\n')
    out.close()
    log.info(f'Wrote {i} lines to {out.name}')


if '__main__' == __name__:
    main()
