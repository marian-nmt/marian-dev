#!/usr/bin/env bash

# DO NOT call this script directly (unless you know what you are doing).
#    Use the build.sh script instead.
# this script builds pymarian wheels for multiple python versions
# it uses mamba to create python environments and builds the wheels
# it also creates manylinux wheels using auditwheel

set -eu
MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MARIAN_ROOT="$( cd "$MYDIR/../.." && pwd )"
# assume this directory is mounted in the docker container
cd $MARIAN_ROOT

#MKL is not in docker image
# yum-config-manager --add-repo https://yum.repos.intel.com/setup/intelproducts.repo
yum-config-manager --add-repo https://yum.repos.intel.com/mkl/setup/intel-mkl.repo
rpm --import https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
yum install -y intel-mkl-64bit-2020.4-912

# TODO: build a docker image with MKL and mamba installed

COMPILE_CUDA=1
PY_VERSIONS="$(echo 3.{12,11,10,9,8})"

# quick testing: compile for only one version and for CPU only
#COMPILE_CUDA=0
#PY_VERSIONS="3.10"

# GLIBC we use for compiling marian should be compatible for newer platforms
# So we use an old GLIBC that works (e.g. 2.17), thus ensuring maximum compatibility
PY_PLATFORM="manylinux_2_17_x86_64"   #  GLIBC must be 2.17 (or older) for this platform
echo "$(ldd --version | head -1); platform=$PY_PLATFORM"
which mamba >& /dev/null || {
    name=Miniforge3-$(uname)-$(uname -m).sh
    mambadir=tmp/mamba-$(uname)-$(uname -m)
    mkdir -p tmp/
    [[ -s $mambadir/bin/activate ]] || {
        [[ -s $name ]] || {
            rm -f $name.tmp
            wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/$name" -O tmp/$name.tmp \
                && mv tmp/$name{.tmp,}
        }
        bash tmp/$name -b -u -p $mambadir/
        $mambadir/bin/mamba init bash
    }
    source $mambadir/etc/profile.d/mamba.sh
    source $mambadir/bin/activate
}

# check if mamba is available
which mamba || {
    echo "mamba not found. Exiting."
    exit 1
}

# create environment for each version

for v in $PY_VERSIONS; do
    mamba env list | grep -q "^py${v}" || {
        echo "Creating python $v environment"
        mamba create -q -y -n py${v} python=${v}
   }
done

# stack all environments
for v in $PY_VERSIONS; do mamba activate py${v} --stack; done
# check if all python versions are available
for v in $PY_VERSIONS; do which python$v; done


# Build as usual
build_dir=$MARIAN_ROOT/build-pymarian
fresh_build=1
if [[ $fresh_build -eq 1 && -d $build_dir ]]; then
    backup_dir=$build_dir.$(date +%y%m%d%H%M%S)
    echo "Moving existing build directory to $backup_dir"
    mv $build_dir $backup_dir
fi

mkdir -p $build_dir
cd $build_dir

#CMAKE_FLAGS="-DPYMARIAN=on -DCMAKE_BUILD_TYPE=Release -DUSE_STATIC_LIBS=on -DUSE_FBGEMM=on"
CMAKE_FLAGS="-DPYMARIAN=on -DCMAKE_BUILD_TYPE=Slim -DUSE_STATIC_LIBS=on -DUSE_FBGEMM=on"
# for cuda support
if [[ $COMPILE_CUDA -eq 1 ]]; then
    CMAKE_FLAGS+=" -DCOMPILE_CUDA=on -DCOMPILE_PASCAL=ON -DCOMPILE_VOLTA=ON -DCOMPILE_TURING=ON -DCOMPILE_AMPERE=ON -DCOMPILE_AMPERE_RTX=ON"
else
    CMAKE_FLAGS+=" -DCOMPILE_CUDA=off -DCOMPILE_CPU=on"
fi

cmake .. $CMAKE_FLAGS
make -j
ls -lh pymarian*.whl

echo "=== Generating manylinux wheels ==="
# make the wheels manylinux compatible
auditwheel repair --plat $PY_PLATFORM *.whl -w manylinux/
ls -lh manylinux/

echo "=== Done ==="
