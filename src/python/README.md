Stuff required to build module (on Linux)
```
python3 -m venv ./venv
source ./venv/bin/activate
python -mpip install sentencepiece scikit-build pybind11
```

Stuff required to run windowed demo
```
python -mpip install pyqt5 sacremoses git+https://github.com/mediacloud/sentence-splitter
```

Build the module (CPU version)
```
# Standard build
python setup.py build -j16 install

# Advanced: using a specific compiler (gcc 9), build CPU only, enable Debug info
CC=gcc-9 CXX=g++-9  python setup.py build -j24 install -- -DCOMPILE_CUDA=off -DCMAKE_BUILD_TYPE=Debug
```

```
echo "Hello World." | python test_translate.py '--config /home/marcinjd/MTMA/decoder.yml' /home/marcinjd/MTMA/source.spm /home/marcinjd/MTMA/target.spm


model_dir=$HOME/tmp/marian-sample
echo "Hello there. How are you" | python bench/test_translate.py "-m $model_dir/model.bin -v $model_dir/vocab.spm $model_dir/vocab.spm" 
```

## Known issues

1. Using conda environment, and `.../miniconda3/envs/<envname>/bin/../lib/libstdc++.so.6: version 'GLIBCXX_3.4.30' not found`
    Install libstdcxx-ng

    ```bash
    conda install -c conda-forge libstdcxx-ng
    ```