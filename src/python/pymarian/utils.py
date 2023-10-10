#!/usr/bin/env python
#
# This is a python wrapper for marian evaluate command
# created by Thamme Gowda on 2023-09-07
#

import logging as log
from pathlib import Path
from typing import Iterator
import shutil

import requests
from tqdm.auto import tqdm

from .constants import Defaults as D

log.basicConfig(level=log.INFO)
DEBUG_MODE=False



def get_known_model(model_name):
    """Given a known model name, gets the model and vocab paths. This function downloads and extracts files if necessary.
    :param model_name: model name
    :return: model path, vocab path
    """
    assert model_name in D.KNOWN_METRICS, f'Unknown model {model_name}'

    model_url = f'{D.BASE_URL}/{model_name}.tgz'
    local_file = D.CACHE_PATH / f'{model_name}.tgz'
    local_dir = D.CACHE_PATH / model_name
    maybe_download_file(model_url, local_file)
    maybe_extract(local_file, local_dir)
    checkpt_file = list(local_dir.glob('model.*.npz'))
    vocab_file = list(local_dir.glob('vocab*.spm'))
    assert len(checkpt_file) == 1, f'Expected exactly one model file in {local_dir}'
    assert len(vocab_file) == 1, f'Expected exactly one vocab file in {local_dir}'
    checkpt_file = checkpt_file[0]
    vocab_file = vocab_file[0]
    return checkpt_file, vocab_file

def maybe_download_file(url, local_file: Path):
    """Downloads the file if not already downloaded
    :param url: url to download
    :param local_file: local file path
    """
    flag_file = local_file.with_name(local_file.name + '._OK')
    if local_file.exists() and flag_file.exists():
        log.info(f'Using cached file {local_file}')
        return
    log.info(f'Downloading {url} to {local_file}')
    local_file.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        file_size = int(r.headers.get('Content-Length', 0))
        with tqdm.wrapattr(r.raw, "read", total=file_size, desc='Downloading', dynamic_ncols=True) as r_raw:
            with open(local_file, "wb") as f:
                shutil.copyfileobj(r_raw, f)
    flag_file.touch()

def maybe_extract(archive: Path, outdir: Path) -> Path:
    """Extracts the archive to outdir if not already extracted
    :param archive: path to archive file
    :param outdir: output directory
    :return: output directory path
    """
    assert archive.exists(), f'{archive} does not exist'
    flag_file = outdir / '._EXTRACT_OK'
    if not outdir.exists() or not flag_file.exists():
        shutil.rmtree(outdir, ignore_errors=True)
        log.info(f'Extracting {archive} to {outdir}')
        # assumption: root dir in tar matches model name
        shutil.unpack_archive(archive, outdir.parent)
        flag_file.touch()
    return outdir

