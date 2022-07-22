import os
import sys

from skbuild import setup

def getVersion(cuda_version=None):
    with open("../../VERSION", encoding = 'utf-8') as f:
        version = f.read().rstrip().lstrip("v") # gets rid of 'v' prefix in v1.17.5 etc.
    v = version if cuda_version is None else f"{version}+cu{cuda_version.replace('.', '')}"
    print("VERSION IS ", v)
    return v

setup(
    name="pymarian",
    packages=["pymarian"],
    version=getVersion(os.environ.get("CUDA_VERSION")),
    author="Marcin Junczys-Dowmunt",
    author_email="marcinjd@microsoft.com",
    description="A test project using pybind11 and CMake",
    long_description="",
    cmake_source_dir="../..",
    install_requires=[
        "setuptools",
        "pybind11",
        "scikit-build",
    ],
    cmake_args = [
        "-DBUILD_ARCH=x86-64",
        "-DCMAKE_BUILD_TYPE=Slim",
        "-DUSE_STATIC_LIBS=ON",
        "-DCOMPILE_AVX2=OFF",
        "-DCOMPILE_AVX512=OFF",
        "-DCOMPILE_CUDA=OFF",
        "-DUSE_FBGEMM=ON",
        "-DUSE_TCMALLOC=OFF",
        "-DUSE_SENTENCEPIECE=ON",
        "-DGENERATE_MARIAN_INSTALL_TARGETS=OFF",
        f"-DPYTHON_EXECUTABLE={sys.executable}",
    ],
    # cmake_install_target = "pymarian",
    zip_safe=False,
    extras_require={"test": ["pytest>=6.0"]},
    python_requires=">=3.6",

)
