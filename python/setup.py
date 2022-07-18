from glob import glob

from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

__version__="0.0.1"

ext_modules = [
    Pybind11Extension("pymarian",
        glob("pymarian/*.cpp"),
        # Example: passing in the version to the compiled code
        define_macros = [('VERSION_INFO', __version__)],
        ),
]

setup(
    include_dirs=["../src", "../src/3rd_party"],
    ext_modules=ext_modules,
)
