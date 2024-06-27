# PyMarian

* Python bindings to Marian (C++) is using [PyBind11]
* The python package is built using [scikit-build-core](https://github.com/scikit-build/scikit-build-core)


## Install

```bash
# build marian with -DPYMARIAN=on option to create a pymarian wheel
cmake . -Bbuild -DCOMPILE_CUDA=off -DPYMARIAN=on -DCMAKE_BUILD_TYPE=Release
cmake --build build -j       # -j option parallelizes build on all cpu cores
python -m pip install build/pymarian-*.whl
```

The above commands use `python` executable in the PATH to determine Python version for compiling marian native extension. Make sure to have the desired `python` executable in your environment _before_ invoking these cmake commands.

## Python API

Python API is designed to take same argument as marian CLI string.
> NOTE: these APIs are experimental only and not finalized. see `mtapi_server.py` for an example use of Translator API 

**Translator**
```python

# Translator
from pymarian import Translator
cli_string = "..."
translator = Translator(cli_string)

sources = ["sent1" , "sent2" ]
result = translator.translate(sources)
print(result)
```

**Evaluator**
```python
# Evaluator
from pymarian import Evaluator
cli_string = '-m path/to/model.npz -v path/to.vocab.spm path/to.vocab.spm --like comet-qe'
evaluator = Evaluator(cli_str)

data = [
    ["Source1", "Hyp1"],
    ["Source2", "Hyp2"]
]
scores = evaluator.run(data)
for score in scores:
    print(score)
```

## CLI Usage
. `pymarian-evaluate` : CLI to download and use pretrained metrics such as COMETs, COMETOIDs, ChrFoid, and BLEURT
. `pymarian-mtapi` : REST API demo powered by Flask
. `pymarian-qtdemo` : GUI App demo powered by QT 


### `pymarian-eval` 

```bash
$ pymarian-eval -h 
usage: pymarian-eval [-h] [-m MODEL] [-v VOCAB] [-l {comet-qe,bleurt,comet}] [-V] [-] [-t MT_FILE] [-s SRC_FILE] [-r REF_FILE] [-f FIELD [FIELD ...]] [-o OUT] [-a {skip,append,only}] [-w WIDTH] [--debug] [--fp16] [--mini-batch MINI_BATCH] [-d [DEVICES ...] | -c
                     CPU_THREADS] [-ws WORKSPACE] [-pc]

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Model name, or path. Known models: bleurt-20, wmt20-comet-da, wmt20-comet-qe-da, wmt20-comet-qe-da-v2, wmt21-comet-da, wmt21-comet-qe-da, wmt21-comet-qe-mqm, wmt22-comet-da, wmt22-cometkiwi-da, xcomet-xl, xcomet-xxL (default: wmt22-cometkiwi-da)
  -v VOCAB, --vocab VOCAB
                        Vocabulary file (default: None)
  -l {comet-qe,bleurt,comet}, --like {comet-qe,bleurt,comet}
                        Model type. Required if --model is a local file (auto inferred for known models) (default: None)
  -V, --version         show program's version number and exit
  -, --stdin            Read input from stdin. TSV file with following format: QE metrics: "src<tab>mt", Ref based metrics ref: "src<tab>mt<tab>ref" or "mt<tab>ref" (default: False)
  -t MT_FILE, --mt MT_FILE
                        MT output file. Ignored when --stdin (default: None)
  -s SRC_FILE, --src SRC_FILE
                        Source file. Ignored when --stdin (default: None)
  -r REF_FILE, --ref REF_FILE
                        Ref file. Ignored when --stdin (default: None)
  -f FIELD [FIELD ...], --fields FIELD [FIELD ...]
                        Input fields, an ordered sequence of {src, mt, ref} (default: ['src', 'mt', 'ref'])
  -o OUT, --out OUT     output file (default: <_io.TextIOWrapper name='<stdout>' mode='w' encoding='utf-8'>)
  -a {skip,append,only}, --average {skip,append,only}
                        Average segment scores to produce system score. skip=do not output average (default; segment scores only); append=append average at the end; only=output the average only (i.e. system score only) (default: skip)
  -w WIDTH, --width WIDTH
                        Output score width (default: 4)
  --debug               Debug or verbose mode (default: False)
  --fp16                Enable FP16 mode (default: False)
  --mini-batch MINI_BATCH
                        Mini-batch size (default: 16)
  -d [DEVICES ...], --devices [DEVICES ...]
                        GPU device IDs (default: None)
  -c CPU_THREADS, --cpu-threads CPU_THREADS
                        Use CPU threads. 0=use GPU device 0 (default: None)
  -ws WORKSPACE, --workspace WORKSPACE
                        Workspace memory (default: 8000)
  -pc, --print-cmd      Print marian evaluate command and exit (default: False)
  --cache CACHE         Cache directory for storing models (default: $HOME/.cache/marian/metric)

More info at https://github.com/marian-nmt/marian-dev. This CLI is loaded from .../python3.10/site-packages/pymarian/eval.py (version: 1.12.25)

```

**Performance Tuning Tips**:
* For CPU parallelization, `--cpu-threads <n>`
* For GPU parallelization, assuming pymarian was compiled with cuda support, e.g., `--devices 0 1 2 3` to use the specified 4 gpu devices.
* When OOM error: adjust `--mini-batch` argument
* To see full logs from marian, set `--debug`



### `pymarian-mtapi`

Launch server
```bash
# example model: download and extract
wget http://data.statmt.org/romang/marian-regression-tests/models/wngt19.tar.gz 
tar xvf wngt19.tar.gz 

# launch server
pymarian-mtapi -s en -t de "-m wngt19/model.base.npz -v wngt19/en-de.spm wngt19/en-de.spm"
```

Example request from client
 
```bash
URL="http://127.0.0.1:5000/translate"
curl $URL --header "Content-Type: application/json" --request POST --data '[{"text":["Good Morning."]}]'
```

### `pymarian-qtdemo` 
```
pymarian-qtdemo
```

## Code Formatting

```bash

pip install black isort
isort .
black .
cd src/python
```

## Run Tests

```bash
# install pytest if necessary
python -m pip install pytest

# run tests in quiet mode
python -m pytest src/python/tests/regression

# or, add -s to see STDOUT/STDERR from tests
python -m pytest -s src/python/tests/regression

```

## Release Instructions

### Building Pymarian for Multiple Python Versions

Our CMake scripts detects `python3.*` available in PATH and builds pymarian for each.
To support a specific version of python, make the `python3.x` executable available in PATH prior to running cmake.
This can be achieved by (without conflicts) using `conda` or `mamba`.


```bash
# setup mamba if not already; Note: you may use conda as well
which mamba || {
   name=Miniforge3-$(uname)-$(uname -m).sh
   wget "https://github.com/conda-forge/miniforge/releases/latest/download/$name" \
      && bash $name -b -p ~/mambaforge && ~/mambaforge/bin/mamba init bash && rm $name
}

# create environment for each version
versions="$(echo 3.{12,11,10,9,8,7})"
for version in $versions; do
   echo "python $version"
   mamba env list | grep -q "^py${version}" || mamba create -q -y -n py${version} python=${version}
done

# stack all environments
for version in $versions; do mamba activate py${version} --stack; done
# check if all python versions are available
for version in $versions; do which python$version; done


# Build as usual
cmake . -B build -DCOMPILE_CUDA=off -DPYMARIAN=on
cmake --build build -j
ls build/pymarian*.whl
```

### Upload to PyPI
```bash
twine upload -r testpypi build/*.whl

twine upload -r pypi build/*.whl
```

__Initial Setup:__ create `~/.pypirc` with following:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
repository: https://upload.pypi.org/legacy/
username:__token__
password:<token>

[testpypi]
repository: https://test.pypi.org/legacy/
username:__token__
password:<token>
```
Obtain token from https://pypi.org/manage/account/ 



## Known issues

1. In conda or mamba environment, if you see  `.../miniconda3/envs/<envname>/bin/../lib/libstdc++.so.6: version 'GLIBCXX_3.4.30' not found` error,
    install libstdcxx-ng

    ```bash
    conda install -c conda-forge libstdcxx-ng
    ```




