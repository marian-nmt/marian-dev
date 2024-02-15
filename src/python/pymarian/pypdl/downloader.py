import copy
import logging
import time
from pathlib import Path
from threading import Event
from typing import Dict

import requests

MEGABYTE = 1048576


class BasicDownloader:
    """Base downloader class."""

    def __init__(self, interrupt: Event):
        self.curr = 0  # Downloaded size in bytes (current size)
        self.completed = False
        self.id = 0
        self.interrupt = interrupt
        self.downloaded = 0

    def download(self, url: str, path: str, mode: str, **kwargs) -> None:
        """Download data in chunks."""
        try:
            with open(path, mode) as file, requests.get(url, stream=True, **kwargs) as response:
                for chunk in response.iter_content(MEGABYTE):
                    file.write(chunk)
                    self.curr += len(chunk)
                    self.downloaded += len(chunk)

                    if self.interrupt.is_set():
                        break

        except Exception as e:
            self.interrupt.set()
            time.sleep(1)
            logging.error("(Thread: %d) [%s: %s]", self.id, type(e).__name__, e)


class Simpledown(BasicDownloader):
    """Class for downloading the whole file in a single segment."""

    def __init__(
        self,
        url: str,
        file_path: str,
        interrupt: Event,
        **kwargs,
    ):
        super().__init__(interrupt)
        self.url = url
        self.file_path = file_path
        self.kwargs = kwargs

    def worker(self) -> None:
        self.download(self.url, self.file_path, mode="wb", **self.kwargs)
        self.completed = True


class Multidown(BasicDownloader):
    """Class for downloading a specific segment of the file."""

    def __init__(
        self,
        segement_table: Dict,
        segment_id: int,
        interrupt: Event,
        **kwargs,
    ):
        super().__init__(interrupt)
        self.id = segment_id
        self.segement_table = segement_table
        self.kwargs = kwargs

    def worker(self) -> None:
        url = self.segement_table["url"]
        segment_path = Path(self.segement_table[self.id]["segment_path"])
        start = self.segement_table[self.id]["start"]
        end = self.segement_table[self.id]["end"]
        size = self.segement_table[self.id]["segment_size"]

        if segment_path.exists():
            downloaded_size = segment_path.stat().st_size
            if downloaded_size > size:
                segment_path.unlink()
            else:
                self.curr = downloaded_size

        if self.curr < size:
            start = start + self.curr
            kwargs = copy.deepcopy(self.kwargs)  # since used by others
            kwargs.setdefault("headers", {}).update({"range": f"bytes={start}-{end}"})
            self.download(url, segment_path, "ab", **kwargs)

        if self.curr == size:
            self.completed = True
