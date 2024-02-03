# PyMarian

* Python bindings to Marian (C++) is using [PyBind11]
* The python package is built using [scikit-build-core](https://github.com/scikit-build/scikit-build-core)


## Install

```bash
# get source code
git clone https://github.com/marian-nmt/marian-dev
cd marian-dev 

# build marian with -DPYMARIAN=on option to create a pymarian wheel
cmake . -Bbuild -DCOMPILE_CUDA=off -DPYMARIAN=on -DCMAKE_BUILD_TYPE=Release
cmake --build build -j       # -j option parallelizes build on all cpu cores
python -m pip install build/pymarian-*.whl
```

Since the above commands uses `python` executable in the PATH to determine Python version to compile marian native extension, make sure to have the desired `python` executable in your environment _before_ invoking these commands.

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


### `pymarian-evaluate` 

```bash
$ pymarian-evaluate -h
usage: pymarian-evaluate [-h] [-m MODEL] [--stdin] [-t MT_FILE] [-s SRC_FILE] [-r REF_FILE] [-o OUT] [-a {skip,append,only}] [-w WIDTH] [--debug] [--mini-batch MINI_BATCH] [-d [DEVICES ...] | -c
                         CPU_THREADS] [-ws WORKSPACE] [--backend {subprocess,pymarian}]

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Model name, or path. Known models=['cometoid22-wmt21', 'cometoid22-wmt22', 'cometoid22-wmt23', 'chrfoid-wmt23', 'comet20-da-qe', 'bleurt20', 'comet20-da'] (default:
                        cometoid22-wmt22)
  --stdin               Read input from stdin. TSV file with following format: QE metrics: "src<tab>mt", Comet with ref: "src<tab>ref<tab>; or BLEURT: "ref<tab>mt" (default: False)
  -t MT_FILE, --mt MT_FILE
                        MT output file. Ignored when --stdin. (default: None)
  -s SRC_FILE, --src SRC_FILE
                        Source file. Ignored when --stdin (default: None)
  -r REF_FILE, --ref REF_FILE
                        Ref file. Ignored when --stdin (default: None)
  -o OUT, --out OUT     output file. Default stdout (default: <_io.TextIOWrapper name='<stdout>' mode='w' encoding='utf-8'>)
  -a {skip,append,only}, --average {skip,append,only}
                        Average segment scores to produce system score. skip=do not output average (default; segment scores only); append=append average at the end; only=output the average only
                        (i.e system score only) (default: skip)
  -w WIDTH, --width WIDTH
                        Output score width (default: 4)
  --debug               Verbose output (default: False)
  --mini-batch MINI_BATCH
                        Mini-batch size (default: 16)
  -d [DEVICES ...], --devices [DEVICES ...]
                        GPU device IDs (default: None)
  -c CPU_THREADS, --cpu-threads CPU_THREADS
                        Use CPU threads. 0=use gpu device 0 (default: None)
  -ws WORKSPACE, --workspace WORKSPACE
                        Workspace memory (default: 8000)
  --backend {subprocess,pymarian}
                        Marian backend interface. subprocess looks for marian binary in PATH. pymarian is a pybind wrapper (default: pymarian)
```

**Performance Tuning Tips**:
* For CPU parallelization, `--cpu-threads <n>`
* For GPU parallelization, assuming pymarian was compiled with cuda support, e.g., `--devices 0 1 2 3` to use the specified 4 gpu devices.
* When OOM error: adjust `--mini-batch` argument
* To see full logs from marian, set `--debug`


*Example Usage*
```bash
# download sample dataset
langs=en-ru
prefix=tmp.$langs
teset=wmt21/systems
sysname=Online-B
sacrebleu -t $teset -l $langs --echo src > $prefix.src
sacrebleu -t $teset -l $langs --echo ref > $prefix.ref
sacrebleu -t $teset -l $langs --echo $sysname > $prefix.mt

# chrfoid
paste $prefix.{src,mt} | head | pymarian-evaluate --stdin -m chrfoid-wmt23 

# cometoid22-wmt{21,22,23}
paste $prefix.{src,mt} | head | pymarian-evaluate --stdin -m cometoid22-wmt22

# bleurt20
paste $prefix.{ref,mt} | head | pymarian-evaluate --stdin  -m bleurt20 --debug

# FIXME: comet20-da-qe and comet20-da appear to be broken 
# comet20-da-qe
paste $prefix.{src,mt} | head | pymarian-evaluate --stdin -m comet20-da-qe
# comet20-da
paste $prefix.{src,mt,ref} | pymarian-evaluate  -m comet20-da 

```

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

## Run Tests

```bash
# install pytest if necessary
python -m pip install pytest 

# run tests in quiet mode
python -m pytest src/python/tests/

# or, add -s to see STDOUT/STDERR from tests
python -m pytest -s src/python/tests/

```


## Known issues
   
1. In conda or mamba environment, if you see  `.../miniconda3/envs/<envname>/bin/../lib/libstdc++.so.6: version 'GLIBCXX_3.4.30' not found` error,
    install libstdcxx-ng

    ```bash
    conda install -c conda-forge libstdcxx-ng
    ```




