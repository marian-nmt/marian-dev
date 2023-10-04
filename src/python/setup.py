import os
import sys
from pathlib import Path


root_dir = Path(__file__).parent.parent.parent.absolute()   # root/src/python/<you-arehere>
version_file = root_dir / "VERSION"
assert version_file.exists(), f"Version file {version_file} does not exist"

def get_version(cuda_version=None):
    version = version_file.read_text().rstrip().lstrip("v")  # gets rid of 'v' prefix in v1.17.5 etc.
    version = version if not cuda_version else f"{version}+cu{cuda_version.replace('.', '')}"
    print("Marian/module version is ", version)
    return version


try:
    from skbuild import setup
except ImportError:
    print("ERROR: skbuild not found. Please install scikit-build first:\n\tpip install scikit-build", file=sys.stderr)
    exit(1)

#BUILD_TYPE="Slim"
BUILD_TYPE="Debug"

setup(
    name="pymarian",
    py_modules=["pymarian"],
    version=get_version(os.environ.get("CUDA_VERSION")),
    author="Marcin Junczys-Dowmunt",
    author_email="marcinjd@microsoft.com",
    description="Python bindings for Marian NMT",
    long_description=(root_dir / "README.md").read_text(),
    long_description_content_type="text/markdown",
    cmake_source_dir=str(root_dir),
    setup_requires=[
        "setuptools",
        "pybind11",
        "scikit-build",
    ],
    cmake_args = [
        "-DBUILD_ARCH=x86-64",
        f"-DCMAKE_BUILD_TYPE={BUILD_TYPE}",
        "-DUSE_STATIC_LIBS=ON",
        "-DCOMPILE_AVX2=ON",
        "-DCOMPILE_AVX512=OFF",
        "-DCOMPILE_CUDA=OFF",
        "-DUSE_FBGEMM=ON",
        "-DUSE_TCMALLOC=OFF",
        "-DUSE_SENTENCEPIECE=ON",
        "-DGENERATE_MARIAN_INSTALL_TARGETS=OFF",
        f"-DPYTHON_EXECUTABLE={sys.executable}",
    ],
    extra_compile_args=["-g"],
    # cmake_install_target = "pymarian",
    zip_safe=False,
    extras_require={"test": ["pytest>=6.0"]},
    python_requires=">=3.6",
)
