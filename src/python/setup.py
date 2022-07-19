import os
import sys

from skbuild import setup

setup(
    name="pymarian",
    version="0.0.1",
    author="Marcin Junczys-Dowmunt",
    author_email="marcinjd@microsoft.com",
    description="A test project using pybind11 and CMake",
    long_description="",
    cmake_source_dir="../..",
    cmake_args = [
        f"-DUSE_SENTENCEPIECE=ON",
        f"-DCOMPILE_CUDA=ON",
        f"-DUSE_FBGEMM=ON",
        f"-DUSE_TCMALLOC=OFF",
        f"-DPYTHON_EXECUTABLE={sys.executable}",
        f"-DCMAKE_BUILD_TYPE=Release",
    ],
    zip_safe=False,
    extras_require={"test": ["pytest>=6.0"]},
    python_requires=">=3.6",
)
