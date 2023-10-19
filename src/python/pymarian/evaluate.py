#!/usr/bin/env python
#
# This is a python wrapper for marian evaluate command
# created by Thamme Gowda on 2023-09-07
#
import sys
import argparse
import logging as log
from pathlib import Path
import subprocess
import threading
import itertools
from typing import Iterator, Optional, List, Union, Tuple
import shutil


from .constants import Defaults as D
from .utils import get_known_model

log.basicConfig(level=log.INFO)
DEBUG_MODE=False


def copy_lines_to_stdin(proc, lines: Iterator[str]):
    """Write data to subproc stdin. Note: run this on another thread to avoid deadlock
    This function reads streams, and write them as TSV record to the stdin of the sub process.
    :param proc: subprocess object to write to
    """

    for line in lines:
        #line = line.rstrip('\n') + '\n'
        proc.stdin.write(line)
    proc.stdin.flush()
    proc.stdin.close()   # close stdin to signal end of input


def marian_evaluate(model: Path, input_lines: Iterator[str],
                vocab_file:Path=None, devices:Optional[List[int]]=None,
                width=D.FLOAT_PRECISION, mini_batch=D.MINI_BATCH, like=D.DEF_SCHEMA,
                maxi_batch=D.MAXI_BATCH, workspace=D.WORKSPACE,
                max_length=D.MAX_LENGTH, cpu_threads=0, average:str='skip', backend='subprocess'
                ) -> Iterator[Union[float, Tuple[float, float]]]:
    """Run marian evaluate, write input and and read scores
    Depending on the `model` argument, either a single score or a tuple of scores is returned per input line.
    :param model: path to model file, or directory containing model.npz.best-embed.npz
    :param vocab: path to vocabulary file (optional; if not given, assumed to be in the same directory as the model)
    :param devices: list of GPU devices to use (optional; if not given, decision is let to marian process)
    :param width: float precision
    :param mini_batch: mini-batch size (default: 16)
    :param like: marian embedding model like (default: comet-qe)
    :param: cpu_threads : number of CPU threads to use (default: 0)
    :param: average: average segment scores to produce system score.
        skip=do not output average (default; segment scores only);
        append=append average at the end;
        only=output the average only (i.e system score only)
    :param: backend: subprocess or pymarian
    :return: iterator over scores.
    """

    assert model.exists()
    if model.is_dir():
        model_dir = model
        _model_files = list(model.glob("*.npz"))
        assert len(_model_files) == 1, f'Expected exactly one model file in {model_dir}'
        model_file = _model_files[0]
    else:
        assert model.is_file()
        model_dir = model.parent
        model_file = model
    if not vocab_file:
        _vocab_files = list(model_dir.glob('*.spm'))
        assert len(_vocab_files) == 1, f'Expected exactly one vocab file in {model_dir}'
        vocab_file = _vocab_files[0]

    assert model_file.exists(), f'{model_file} does not exist'
    assert vocab_file.exists(), f'{vocab_file} does not exist'

    n_inputs = len(D.KNOWN_SCHEMA[like].split('+'))
    vocabs = [vocab_file] * n_inputs
    kwargs = dict(
        model=model_file,
        vocabs=vocabs,
        devices=devices,
        width=width,
        like=like,
        mini_batch=mini_batch,
        maxi_batch=maxi_batch,
        max_length=max_length,
        max_length_crop=True,
        workspace=workspace,    # negative memory => relative to total memory
        cpu_threads=cpu_threads,
        average=average
    )

    cmd_line = []
    for key, val in kwargs.items():
        if val is None:   # ignore this key / flag
            continue
        cmd_line.append(f"--{key.replace('_', '-')}")
        if val is True:   # boolean flag
            cmd_line.append('true')
        elif val is False:
            cmd_line.append('false')

        elif isinstance(val, (list, tuple)):
            cmd_line.extend(str(v) for v in val)
        else:
            cmd_line.append(str(val))
    if not DEBUG_MODE:
        cmd_line.append('--quiet')
    if backend == 'subprocess':
        return subprocess_evaluate(cmd_line, input_lines)
    elif backend == 'pymarian':
        cmd_line = ' '.join(cmd_line)
        batch_size = mini_batch * maxi_batch
        return pymarian_evaluate(cmd_line, input_lines, batch_size=batch_size)
    else:
        raise ValueError(f'Unknown backend {backend}')

    

def pymarian_evaluate(cmd_line: str, input_lines: Iterator[str], 
                      batch_size=int(D.MINI_BATCH * D.MAXI_BATCH)):
    try:
        from pymarian import Evaluator
    except:
        raise ImportError('pymarian is not installed. Please install it and rerun')

    log.info(f'Marian CLI::\n\t{cmd_line}')
    evaluator = Evaluator(cmd_line)
    
    lines = (line.rstrip('\n').split('\t') for line in input_lines)
    # NOTE: pymarian doesnt support iterator input yet; so mini batching here
    # TODO: support iterator input
    def make_mini_batches(lines, batch_size=batch_size):
        assert batch_size > 0
        while True:
            chunk = list(itertools.islice(lines, batch_size))
            if not chunk:
                return
            yield chunk
    for batch in make_mini_batches(lines):
        scores = evaluator.run(batch)
        assert len(scores) == len(batch)
        for score in scores:
            yield score


def subprocess_evaluate(cmd_line: List[str], input_lines: Iterator[str]):
    assert isinstance(cmd_line, list)
    marian_bin_path = shutil.which('marian')
    if marian_bin_path is None:
        raise FileNotFoundError('marian binary not found in PATH. Please add it and rerun')
    cmd_line = [marian_bin_path, 'evaluate'] +  cmd_line

    proc = None
    try:
        proc = subprocess.Popen(cmd_line, shell=False, stdout=subprocess.PIPE, stdin=subprocess.PIPE,
                                stderr=sys.stderr, text=True, encoding='utf8', errors='replace')
        log.info(f'Running command: {" ".join(cmd_line)}')
        copy_thread = threading.Thread(target=copy_lines_to_stdin, args=(proc, input_lines))

        copy_thread.start()
        # read output and yield scores
        for line in proc.stdout:
            line = line.rstrip()
            if ' ' in line:
                yield tuple(float(x) for x in line.split(' '))
            else:
                yield float(line)

        # wait for copy thread to finish
        copy_thread.join()
        #proc.stdin.close()
        returncode = proc.wait()
        if returncode != 0:
            raise RuntimeError(f'Process exited with code {returncode}')
    finally:
        if proc is not None and proc.returncode is None:
            log.warning(f'Killing process {proc.pid}')
            proc.kill()


def parse_args():
    parser = argparse.ArgumentParser(
         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-m', '--model', help=f'Model name, or path. Known models={list(D.KNOWN_METRICS.keys())}',
                        default=D.DEF_MODEL, type=str)

    parser.add_argument('--stdin', action='store_true',
                        help='Read input from stdin. TSV file with following format: \
                          QE metrics: "src<tab>mt", Comet with ref: "src<tab>ref<tab>; or BLEURT: "ref<tab>mt"')
    parser.add_argument('-t', '--mt',  dest='mt_file', help='MT output file. Ignored when --stdin.', type=Path)
    parser.add_argument('-s', '--src', dest='src_file', help='Source file. Ignored when --stdin', type=Path)
    parser.add_argument('-r', '--ref', dest='ref_file', help='Ref file. Ignored when --stdin', type=Path)
    parser.add_argument('-o', '--out', default=sys.stdout, help='output file. Default stdout', type=argparse.FileType('w'))
    parser.add_argument('-a', '--average', choices=('skip','append', 'only'),  default='skip',
                        help='Average segment scores to produce system score.'
                        ' skip=do not output average (default; segment scores only);'
                        ' append=append average at the end; '
                        ' only=output the average only (i.e system score only)')

    parser.add_argument('-w', '--width', default=4, help='Output score width', type=int)
    parser.add_argument('--debug', help='Verbose output', action='store_true')
    parser.add_argument('--mini-batch', default=16, help='Mini-batch size', type=int)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-d', '--devices', nargs='*', type=int,
                       help='GPU device IDs')
    group.add_argument('-c', '--cpu-threads', default=None, type=int,
                       help='Use CPU threads. 0=use gpu device 0')
    parser.add_argument('-ws', '--workspace', default=8000, help='Workspace memory', type=int)
    parser.add_argument('--backend', default='pymarian', choices=['subprocess', 'pymarian'], 
                        help='Marian backend interface. subprocess looks for marian binary in PATH. pymarian is a pybind wrapper')

    args = parser.parse_args()
    return vars(args)


def read_input(args, model_id, schema=None):
    model_schema = D.KNOWN_METRICS.get(model_id, schema or D.DEF_SCHEMA)
    input_schema = D.KNOWN_SCHEMA[model_schema]
    n_inputs = len(input_schema.split('+'))
    if args.pop('stdin'):
        del args['mt_file']
        del args['src_file']
        del args['ref_file']
        return sys.stdin

    n_inputs = len(input_schema.split('+'))
    mt_file = args.pop('mt_file')
    src_file = args.pop('src_file')
    ref_file = args.pop('ref_file')
    assert mt_file.exists(), f'{mt_file} does not exist'
    if 'src' in input_schema:
        assert src_file, f'Source file is required for metric {model_id}'
        assert src_file.exists(), f'{src_file} does not exist'
    if 'ref' in input_schema:
        assert ref_file, f'Reference file is required for metric {model_id}'
        assert ref_file.exists(), f'{ref_file} does not exist'
    if input_schema == 'src+mt':
        input_lines = itertools.zip_longest(open(src_file), open(mt_file))
    elif input_schema == 'src+ref+mt':
        input_lines = itertools.zip_longest(open(src_file), open(ref_file), open(mt_file))
    elif input_schema == 'src+mt+ref':
        input_lines = itertools.zip_longest(open(src_file), open(mt_file), open(ref_file))
    elif input_schema == 'ref+mt':
        input_lines = itertools.zip_longest(open(ref_file), open(mt_file))
    else:
        raise ValueError(f'Unknown schema {input_schema}')

    def _validate_and_join():
        for row in input_lines:
            assert len(row) == n_inputs, f'Expected {n_inputs} columns, but got {len(row)}'
            for col in row:
                assert col is not None, f'Expected {n_inputs} columns, but got {len(row)}'
            yield '\t'.join(row)
    return _validate_and_join()

def main(**args):
    args = args or parse_args()
    if args.pop('debug'):
        log.getLogger().setLevel(log.DEBUG)
        global DEBUG_MODE
        DEBUG_MODE=True
        log.debug(args)

    model_id = args.pop('model')
    if model_id.lower() in D.KNOWN_METRICS:
        model_path, vocab = get_known_model(model_id.lower())
        log.info(f'{model_id} --> {model_path}')
    else:
        model_path, vocab = Path(model_id), None
    assert model_path.exists(), f'{model_path} does not exist. Known models are {list(D.KNOWN_METRICS.keys())}'
    args['model'] = model_path
    args['vocab_file'] = vocab


    args['input_lines'] = read_input(args, model_id=model_id)
    args['like'] = D.KNOWN_METRICS.get(model_id, D.DEF_SCHEMA)
    out = args.pop('out')
    width = args.pop('width', D.FLOAT_PRECISION)
    scores = marian_evaluate(**args)
    for i, score in enumerate(scores, start=1):
        if isinstance(score, (tuple, list)):
            score = score[0]  # the first score
        out.write(f'{score:.{width}f}\n')
    out.close()

    log.info(f'Wrote {i} lines to {out.name}')

if '__main__' == __name__:
    main()
