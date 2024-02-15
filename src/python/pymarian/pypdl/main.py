import logging
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import Callable, Optional, Union

import requests
from tqdm import tqdm

from .downloader import Multidown, Simpledown
from .utils import (
    AutoShutdownFuture,
    FileValidator,
    combine_files,
    create_segment_table,
    get_filepath,
    seconds_to_hms,
    to_mb,
)

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)


class Downloader:
    """
    A multi-threaded file downloader that supports progress tracking, retries, pause/resume functionality etc.

    Keyword Arguments:
        params (dict, optional): A dictionary, list of tuples or bytes to send as a query string. Default is None.
        allow_redirects (bool, optional): A Boolean to enable/disable redirection. Default is True.
        auth (tuple, optional): A tuple to enable a certain HTTP authentication. Default is None.
        cert (str or tuple, optional): A String or Tuple specifying a cert file or key. Default is None.
        cookies (dict, optional): A dictionary of cookies to send to the specified url. Default is None.
        headers (dict, optional): A dictionary of HTTP headers to send to the specified url. Default is None.
        proxies (dict, optional): A dictionary of the protocol to the proxy url. Default is None.
        timeout (number or tuple, optional): A number, or a tuple, indicating how many seconds to wait for the client to make a connection and/or send a response. Default is 10 seconds.
        verify (bool or str, optional): A Boolean or a String indication to verify the servers TLS certificate or not. Default is True.
    """

    def __init__(self, **kwargs):
        self._pool = None  # ThreadPoolExecutor, initialized in _downloader
        self._workers = []
        self._interrupt = Event()
        self._stop = False
        self._kwargs = {"timeout": 10, "allow_redirects": True}  # request module kwargs
        self._kwargs.update(kwargs)

        # public attributes
        self.size = None
        self.progress = 0
        self.speed = 0
        self.time_spent = 0
        self.current_size = 0
        self.eta = "99:59:59"
        self.remaining = None
        self.failed = False
        self.completed = False

    def _display(self, dynamic_print):
        dynamic_print.update(self.current_size - dynamic_print.n)

    def _calc_values(self, recent_queue, interval):
        self.current_size = sum(worker.curr for worker in self._workers)

        # Speed calculation
        recent_queue.append(sum(worker.downloaded for worker in self._workers))
        non_zero_list = [to_mb(value) for value in recent_queue if value]
        if len(non_zero_list) < 1:
            self.speed = 0
        elif len(non_zero_list) == 1:
            self.speed = non_zero_list[0] / interval
        else:
            diff = [b - a for a, b in zip(non_zero_list, non_zero_list[1:])]
            self.speed = (sum(diff) / len(diff)) / interval

        if self.size:
            self.progress = int(100 * self.current_size / self.size)
            self.remaining = to_mb(self.size - self.current_size)

            if self.speed:
                self.eta = seconds_to_hms(self.remaining / self.speed)
            else:
                self.eta = "99:59:59"

    def _single_thread(self, url, file_path):
        sd = Simpledown(url, file_path, self._interrupt, **self._kwargs)
        self._workers.append(sd)
        self._pool.submit(sd.worker)

    def _multi_thread(self, segments, segement_table):
        for segment in range(segments):
            md = Multidown(
                segement_table,
                segment,
                self._interrupt,
                **self._kwargs,
            )
            self._workers.append(md)
            self._pool.submit(md.worker)

    def _get_header(self, url):
        kwargs = self._kwargs.copy()
        kwargs.pop("params", None)
        response = requests.head(url, **kwargs)

        if response.status_code != 200:
            self._interrupt.set()
            raise ConnectionError(f"Server Returned: {response.reason}({response.status_code}), Invalid URL")

        return response.headers

    def _get_info(self, url, file_path, multithread, etag):
        header = self._get_header(url)
        file_path = get_filepath(url, header, file_path)

        if size := int(header.get("Content-Length", 0)):
            self.size = size

        return file_path, multithread, etag

    def _downloader(self, url, file_path, segments, display, multithread, etag):
        start_time = time.time()

        file_path, multithread, etag = self._get_info(url, file_path, multithread, etag)

        if multithread:
            segment_table = create_segment_table(url, file_path, segments, self.size, etag)
            segments = segment_table["segments"]
            self._pool = ThreadPoolExecutor(max_workers=segments)
            self._multi_thread(segments, segment_table)
        else:
            self._pool = ThreadPoolExecutor(max_workers=1)
            self._single_thread(url, file_path)

        interval = 0.15
        recent_queue = deque([0] * 12, maxlen=12)
        download_mode = "Multi-Threaded" if multithread else "Single-Threaded"

        with tqdm(total=self.size, desc=f"Downloading ({download_mode})", dynamic_ncols=True, unit='B', unit_scale=True, miniters=1) as dynamic_print:
            while True:
                status = sum(worker.completed for worker in self._workers)
                self._calc_values(recent_queue, interval)

                if display:
                    self._display(dynamic_print)

                if self._interrupt.is_set():
                    self.time_spent = time.time() - start_time
                    return None

                if status == len(self._workers):
                    if multithread:
                        combine_files(file_path, segments)
                    self.completed = True
                    self.time_spent = time.time() - start_time
                    return FileValidator(file_path)

                time.sleep(interval)

    def stop(self) -> None:
        """Stop the download process."""
        self._interrupt.set()
        self._stop = True
        time.sleep(1)  # wait for threads

    def start(
        self,
        url: str,
        file_path: Optional[str] = None,
        segments: int = 10,
        display: bool = True,
        multithread: bool = True,
        block: bool = True,
        retries: int = 0,
        mirror_func: Optional[Callable[[], str]] = None,
        etag: bool = True,
    ) -> Union[AutoShutdownFuture, FileValidator, None]:
        """
        Start the download process.

        Parameters:
            url (str): The URL to download from.
            file_path (str, Optional): The path to save the downloaded file. If not provided, the file is saved in the current working directory.
                If `file_path` is a directory, the file is saved in that directory. If `file_path` is a file name, the file is saved with that name.
            segments (int, Optional): The number of segments to divide the file into for multi-threaded download. Default is 10.
            display (bool, Optional): Whether to display download progress and other messages. Default is True.
            multithread (bool, Optional): Whether to use multi-threaded download. Default is True.
            block (bool, Optional): Whether to block the function until the download is complete. Default is True.
            retries (int, Optional): The number of times to retry the download if it fails. Default is 0.
            mirror_func (Callable[[], str], Optional): A function that returns a new download URL if the download fails. Default is None.
            etag (bool, Optional): Whether to validate the ETag before resuming downloads. Default is True.

        Returns:
            AutoShutdownFuture: If `block` is False.
            FileValidator: If `block` is True and the download successful.
            None: If `block` is True and the download fails.
        """

        def download():
            for i in range(retries + 1):
                try:
                    _url = mirror_func() if i > 0 and callable(mirror_func) else url
                    if i > 0 and display:
                        logging.info("Retrying... (%d/%d)", i, retries)

                    self.__init__(**self._kwargs)
                    result = self._downloader(_url, file_path, segments, display, multithread, etag)

                    if self._stop or self.completed:
                        if display:
                            print(f"Time elapsed: {seconds_to_hms(self.time_spent)}", file=sys.stderr)
                        return result

                    time.sleep(3)

                except Exception as e:
                    logging.error("(%s) [%s]", e.__class__.__name__, e)

                finally:
                    self._pool.shutdown()

            self.failed = True
            return None

        ex = ThreadPoolExecutor(max_workers=1)
        future = AutoShutdownFuture(ex.submit(download), ex)

        if block:
            result = future.result()
            return result

        return future
