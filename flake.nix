{
  description = "Build marian-nmt's marian-dev with CUDA support";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.simpleFlake {
      name = "marian-nmt";

      inherit self nixpkgs;

      config = {
        allowUnfree = true;  # for cuda
        #cudaSupport = true; # what would this change?
      };

      shell = { pkgs }:
          # nvcc 12.8 supports up to gcc 14, so downgrading to it.
          # See https://discourse.nixos.org/t/how-to-select-a-specific-version-of-gcc-within-a-flake-nix/23372/4
          #
          # NOTE(USE_LOCAL_NCCL): compiling with the bundled nccl requires gcc13Stdenv, downgrade if you would like
          # to use that option.
          (pkgs.mkShell.override {stdenv = pkgs.gcc14Stdenv;}){

          buildInputs = [
            pkgs.cmake
            pkgs.gcc
            pkgs.git
            pkgs.sentencepiece
            pkgs.gperftools # for tcmalloc
            pkgs.cudaPackages.nccl  # NOTE(USE_LOCAL_NCCL): would not be needed for the local option
            pkgs.cudaPackages.libcublas
            pkgs.cudaPackages.libcusparse
            pkgs.cudaPackages.cuda_nvcc
            pkgs.cudaPackages.cuda_cudart
            pkgs.cudaPackages.cudatoolkit
            pkgs.mkl  # optional
          ];
          # Example usage:
          #
          # # Bump workspace memory if trace shows too many repeated reallocations.
          # LD_LIBRARY_PATH=/run/nvidia-libs ../marian score --train-sets /some/train.{en,de} --model /some/model.npz --vocabs /some/vocab.{spm,spm} -d 0 -w 8000 --log-level trace --fp16

          # Some rationale of passed flags:
          #
          # USE_NCCL=OFF: otherwise some nccl-y functions can't be linked, when targeting low archs (sm70-ish?).
          #   Why does it even try to do that, if we disabled those archs? Well, we don't need NCCL for now.
          #
          # BUILD_SHARED_LIBS=true: otherwise the shared MKL libs are not picked up.
          #
          # unset NIX_ENFORCE_NO_NATIVE: otherwise __m128 and other sse/avx things
          # won't be available, due to removal of -march=native (this could be
          # set explicitly).
          #
          shellHook = ''
            mkdir -p build
            cd build
            unset NIX_ENFORCE_NO_NATIVE
            #cmake -DCMAKE_BUILD_TYPE=Release \
            #  -DCOMPILE_KEPLER=OFF -DCOMPILE_MAXWELL=OFF -DCOMPILE_PASCAL=OFF -DCOMPILE_VOLTA=OFF -DCOMPILE_TURING=OFF -DCOMPILE_AMPERE=OFF \
            #  -DCOMPILE_AMPERE_RTX=ON  \
            cmake -DCMAKE_BUILD_TYPE=Release \
              -DBUILD_SHARED_LIBS=true \
              -DUSE_NCCL=ON \
              -DDISABLE_CUBLASLT_DEFAULT_PATH_LOOKUP=OFF \
              ..
            make -j
            # just to get feedback on whan arch/gpu we have included
            TEST_CU=src/CMakeFiles/marian_cuda.dir/tensors/gpu/marian_cuda_generated_hash.cu.o
            [ -e $TEST_CU ] && cuobjdump $TEST_CU
          '';
        };
    };
}

