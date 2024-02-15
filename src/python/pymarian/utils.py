#!/usr/bin/env python
#
# This is a python wrapper for marian evaluate command


import logging as log
import shutil
from pathlib import Path
from typing import List, Tuple

import portalocker
import requests

from .defaults import Defaults
from .pypdl import Downloader

log.basicConfig(level=log.INFO)

DEBUG_MODE = False
PROGRESS_BAR = Defaults.PROGRESS_BAR


class InvalidIDException(ValueError):
    """Invalid model ID exception"""

    pass


def validate_id(id: str) -> bool:
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for c in invalid_chars:
        if c in id:
            raise InvalidIDException(
                f'Invalid model id {id}. It must not contain characters: {invalid_chars}'
            )


def get_model_path(model_name, progress_bar: bool = PROGRESS_BAR) -> Path:
    """Given the name of a (known) model, this function gets its checkpoint path.
    If necessary, this function downloads checkpoint to a local cache directory.

    :param model_name: model name
    :return: checkpoint path
    """
    validate_id(model_name)
    chkpt_url = f'{Defaults.BASE_URL}/{model_name}/model.{model_name}.bin'

    local_dir = Defaults.CACHE_PATH / model_name
    chkpt_local = local_dir / f'model.{model_name}.bin'

    maybe_download_file(chkpt_url, chkpt_local)
    assert chkpt_local.exists(), f'Checkpoint file {chkpt_local} does not exist'
    return chkpt_local


def get_vocab_path(model_name, progress_bar: bool = PROGRESS_BAR) -> Tuple[Path, Path]:
    """Given the name of a (known) model, this function gets its vocabulary path.
    This function downloads vocabulary to a local cache directory, if necessary.

    :param model_name: model name
    :param progress_bar: show progress bar while downloading
    :return: checkpoint path, vocabulary path
    """
    validate_id(model_name)
    local_dir = Defaults.CACHE_PATH / model_name
    vocab_local = local_dir / 'vocab.spm'

    vocab_url = f'{Defaults.BASE_URL}/{model_name}/vocab.spm'
    maybe_download_file(vocab_url, vocab_local, progress_bar=progress_bar)
    assert vocab_local.exists(), f'Vocabulary file {vocab_local} does not exist'
    return vocab_local


def maybe_download_file(url: str, local_file: Path, progress_bar: bool = PROGRESS_BAR):
    """Downloads the file if not already downloaded
    :param url: url to download
    :param local_file: local file path
    :param progress_bar: show progress bar while downloading
    :return: None
    :raises: ValueError if the url is invalid
    """
    lock_file = local_file.with_name('._LOCK_' + local_file.name)
    if local_file.exists() and local_file.stat().st_size > 0:
        log.debug(f'Using cached file {local_file}')
        return

    # check if the url has OK status; avoid creating cache directories when url is invalid due to bad model ID
    if not is_ok_url(url):
        raise ValueError(f'Invalid URL: {url}')

    local_file.parent.mkdir(parents=True, exist_ok=True)
    with portalocker.Lock(lock_file, 'w', timeout=Defaults.FILE_LOCK_TIMEOUT) as fh:
        # check again if it is downloaded by another process while we were waiting for the lock
        if local_file.exists() and local_file.stat().st_size > 0:
            log.debug(f'Using cached file {local_file}')
            return

        # use file lock to avoid race of parallel downloads
        local_file.parent.mkdir(parents=True, exist_ok=True)

        tmp_file = local_file.with_name(local_file.name + '.downloading')
        log.info(f'Downloading {url} to {tmp_file}')
        dl = Downloader()
        dl.start(
            url=url,
            file_path=tmp_file,
            segments=20,
            display=progress_bar,
            multithread=True,
            block=True,
            retries=3,
            mirror_func=None,
            etag=False,
        )

        if dl.completed:
            # move the file to the final location
            if local_file.exists():
                local_file.unlink()
            shutil.move(tmp_file, local_file)


def is_ok_url(url: str) -> bool:
    """Checks if the given url has OK status code by making a HEAD request
    :param url: url
    :return: True if status is OK, False otherwise
    """
    try:
        return requests.head(url).status_code == requests.codes.ok
    except requests.exceptions.RequestException as e:
        log.error(f'Invalid URL: {url}')
        return False


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


def kwargs_to_cli(**kwargs) -> str:
    """Converts kwargs to command line arguments string
    :param kwargs: kwargs
    :return: CLI string
    """
    args = []
    for k, v in kwargs.items():
        if v is None:
            continue  # ignore keys if values are None
        k = k.replace('_', '-')
        args.append(f'--{k}')
        if v == '':
            continue  # only add keys for empty values
        elif isinstance(v, bool):
            args.append("true" if v else "false")
        elif isinstance(v, (list, tuple)):
            args.extend(str(x) for x in v)
        else:
            args.append(f'{v}')
    return ' '.join(args)

