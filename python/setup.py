from glob import glob

from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

__version__="0.0.1"

ext_modules = [
    Pybind11Extension("pymarian",
        glob("pymarian/*.cpp"),
        # Example: passing in the version to the compiled code
        define_macros = [('VERSION_INFO', __version__)],
        extra_objects=[
            "../build/libmarian.a",
            "../build/src/3rd_party/intgemm/libintgemm.a",
            "../build/src/3rd_party/sentencepiece/src/libsentencepiece.a",
            "../build/src/3rd_party/sentencepiece/src/libsentencepiece_train.a",
            # "/opt/intel/mkl/lib/intel64_lin/libmkl_gf_lp64.a",
            # "/opt/intel/mkl/lib/intel64_lin/libmkl_pgi_thread.a",
            # "/opt/intel/mkl/lib/intel64_lin/libmkl_blas95_ilp64.a",
            # "/opt/intel/mkl/lib/intel64_lin/libmkl_gnu_thread.a",
            # "/opt/intel/mkl/lib/intel64_lin/libmkl_lapack95_lp64.a",
            # "/opt/intel/mkl/lib/intel64_lin/libmkl_blacs_intelmpi_ilp64.a",
            # "/opt/intel/mkl/lib/intel64_lin/libmkl_lapack95_ilp64.a",
            # "/opt/intel/mkl/lib/intel64_lin/libmkl_blas95_lp64.a",
            # "/opt/intel/mkl/lib/intel64_lin/libmkl_scalapack_ilp64.a",
            # "/opt/intel/mkl/lib/intel64_lin/libmkl_blacs_openmpi_lp64.a",
            # "/opt/intel/mkl/lib/intel64_lin/libmkl_tbb_thread.a",
            # "/opt/intel/mkl/lib/intel64_lin/libmkl_sequential.a",
            # "/opt/intel/mkl/lib/intel64_lin/libmkl_blacs_sgimpt_ilp64.a",
            # "/opt/intel/mkl/lib/intel64_lin/libmkl_blacs_intelmpi_lp64.a",
            # "/opt/intel/mkl/lib/intel64_lin/libmkl_intel_ilp64.a",
            # "/opt/intel/mkl/lib/intel64_lin/libmkl_scalapack_lp64.a",
            # "/opt/intel/mkl/lib/intel64_lin/libmkl_core.a",
            # "/opt/intel/mkl/lib/intel64_lin/libmkl_cdft_core.a",
            # "/opt/intel/mkl/lib/intel64_lin/libmkl_blacs_sgimpt_lp64.a",
            # "/opt/intel/mkl/lib/intel64_lin/libmkl_blacs_openmpi_ilp64.a",
            # "/opt/intel/mkl/lib/intel64_lin/libmkl_gf_ilp64.a",
            # "/opt/intel/mkl/lib/intel64_lin/libmkl_intel_thread.a",
            # "/opt/intel/mkl/lib/intel64_lin/libmkl_intel_lp64.a"           
        ],
    ),
]

setup(
    include_dirs=["../src", "../src/3rd_party"],
    ext_modules=ext_modules,
)
