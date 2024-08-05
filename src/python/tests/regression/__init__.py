import os

QUIET = os.getenv('MARIAN_QUIET', "").lower() in ("1", "yes", "y", "true", "on")
CPU_THREADS = int(os.getenv('MARIAN_CPU_THREADS', "4"))
WORKSPACE_MEMORY = int(os.getenv('MARIAN_WORKSPACE_MEMORY', "6000"))

EPSILON = 0.0001  # the precision error we afford in float comparison

BASE_ARGS = dict(
    mini_batch=8,
    maxi_batch=64,
    cpu_threads=CPU_THREADS,
    workspace=WORKSPACE_MEMORY,
    quiet=QUIET,
)
