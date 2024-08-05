import logging as log
import os
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path
from typing import List

import pytest

log.basicConfig(level=log.INFO)

DATA_URL = "https://textmt.blob.core.windows.net/www/data/marian-regression-tests/metrics-regression.tgz"
DATA_DIR = Path(__file__).parent.parent / "data" / "metrics-regression"
SELECT_PREFIX = "wmt21-systems.en-de.100"
SYS_DIFF_OK = 0.01
SEG_DIFF_OK = 0.05

N_CPUS = max(os.cpu_count() - 2, 2)

USE_GPU = False
GPU_ARGS = "-d 0 --mini-batch 16"  # --fp16 error margin is too high for bleurt-20
CPU_ARGS = f"--cpu-threads {N_CPUS} --mini-batch 1"
# NOTE: --mini-batch > 1 on CPU deviates scores https://machinetranslation.visualstudio.com/DefaultCollection/Marian/_git/marian-dev/pullRequest/32883#1707853099
BACKEND_ARGS = GPU_ARGS if USE_GPU else CPU_ARGS

src_file = DATA_DIR / f"{SELECT_PREFIX}.src"
ref_file = DATA_DIR / f"{SELECT_PREFIX}.ref"
mt_file = DATA_DIR / f"{SELECT_PREFIX}.mt"


def setup():
    try:
        flag_file = DATA_DIR / ".downloaded"
        if flag_file.exists():
            log.info("Data already downloaded. Setup skipped...")
            return

        DATA_DIR.mkdir(exist_ok=True, parents=True)
        log.info(f"Downloading {DATA_URL} to {DATA_DIR}")
        print("Downloading data package...")
        with urllib.request.urlopen(DATA_URL) as response:
            with tarfile.open(fileobj=response, mode="r|gz") as tar:
                tar.extractall(path=DATA_DIR.parent)

        flag_file.touch()
        log.info("Setup Done.")
    finally:
        if not shutil.which("pymarian-eval"):
            raise FileNotFoundError("pymarian-eval not found in PATH")
        for f in [src_file, ref_file, mt_file]:
            if not f.exists():
                raise FileNotFoundError(f"File {f} not found.")


def compare_scores(tag: str, lhs: List[float], rhs: List[float]):
    assert len(lhs) == len(rhs), f"{tag} :: length mismatch: {len(lhs)} != {len(rhs)}"
    total_diff = sum(abs(l - r) for l, r in zip(lhs, rhs))
    avg_diff = total_diff / len(lhs)

    seg_err_count = 0
    for i, (l, r) in enumerate(zip(lhs, rhs)):
        if abs(l - r) >= SEG_DIFF_OK:
            log.warning(f"{tag} :: line {i}: {l:.4f} != {r:.4f} ({abs(l - r):.4f} > {SEG_DIFF_OK})")
            seg_err_count += 1

    assert avg_diff <= SYS_DIFF_OK, f"{tag} :: avg_diff: {avg_diff:.4f} > {SYS_DIFF_OK:.4f}"
    assert seg_err_count == 0, f"{tag} :: seg_err_count: {seg_err_count:.4f} > 0"


setup()
# auto detect metric names
# metric_names = list(set(f.name.split(".")[-2] for f in DATA_DIR.glob(f"{select_prefix}*.orig")))
# update: No need to run all metric names, select a few
metric_names = ["bleurt-20", "wmt20-comet-qe-da", "wmt22-comet-da", "wmt22-cometkiwi-da"]


@pytest.mark.parametrize("metric_name", metric_names)
def test_pymarian_cli(metric_name):
    orig_file = DATA_DIR / f"{SELECT_PREFIX}.{metric_name}.orig"
    assert orig_file.exists()
    orig_scores = [float(x) for x in orig_file.read_text().splitlines() if x.strip()]

    pymarian_args = f"-a skip -s {src_file} -r {ref_file} -t {mt_file} {BACKEND_ARGS}"
    cmd = f"pymarian-eval -m {metric_name} {pymarian_args} "
    log.info(f"Running: {cmd}")
    output = subprocess.check_output(cmd, shell=True)
    out_lines = output.decode("utf-8").splitlines()
    out_scores = [float(x) for x in out_lines if x.strip()]
    compare_scores(metric_name, orig_scores, out_scores)
