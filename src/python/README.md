# PyMarian

* Python bindings to Marian (C++) is using [PyBind11]
* The python package is built using [scikit-build-core](https://github.com/scikit-build/scikit-build-core)


## Install

```bash
# get source code
git clone https://github.com/marian-nmt/marian-dev
cd marian-dev 

# build and install -- along with optional dependencies for demos
# run this from root of project, i.e., dir with pyproject.toml
pip install -v .[demos]   

# using a specific version of compiler (e.g., gcc-9 g++-9)
CMAKE_ARGS="-DCMAKE_C_COMPILER=gcc-9 -DCMAKE_CXX_COMPILER=g++-9" pip install -v .[demos]

# with CUDA on
CMAKE_ARGS="-DCOMPILE_CUDA=ON" pip install . 

# with a specific version of cuda toolkit, e,g. cuda 11.5
CMAKE_ARGS="-DCOMPILE_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.5" pip install -v .[demos]

# all the above combined 
CMAKE_ARGS="-DCMAKE_C_COMPILER=gcc-9 -DCMAKE_CXX_COMPILER=g++-9 -DCOMPILE_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.1" pip install -v .[demos]

# parallel build
#MAKE_ARGS="-DCMAKE_BUILD_PARALLEL_LEVEL=24"
```


## Python API

Python API is designed to take same argument as marian CLI string.

**Translator**
```python

# Translator
from pymarian import Translator
cli_string = "..."
translator = Translator(cli_string)

sources = ["sent1" , "sent2" ]
result = translator.run(sources)
print(result)
```

**Evalautor**
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
. `pymarian-qtdmo` : GUI App demo powered by QT 


### Pymarian-Evaluate

```bash
$ pymarian-evaluate -h
usage: pymarian-evaluate [-h] [-m MODEL] [--stdin] [-t MT_FILE] [-s SRC_FILE] [-r REF_FILE] [-o OUT] [-a {skip,append,only}] [-w WIDTH]
                         [--debug] [--mini-batch MINI_BATCH] [-d [DEVICES ...] | -c CPU_THREADS] [-ws WORKSPACE]
                         [--backend {subprocess,pymarian}]

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Model name, or path. Known models=['cometoid22-wmt21', 'cometoid22-wmt22', 'cometoid22-wmt23', 'chrfoid-wmt23',
                        'comet20-da-qe', 'bleurt20', 'comet20-da'] (default: cometoid22-wmt22)
  --stdin               Read input from stdin. TSV file with following format: QE metrics: "src<tab>mt", Comet with ref:
                        "src<tab>ref<tab>; or BLEURT: "ref<tab>mt" (default: False)
  -t MT_FILE, --mt MT_FILE
                        MT output file. Ignored when --stdin. (default: None)
  -s SRC_FILE, --src SRC_FILE
                        Source file. Ignored when --stdin (default: None)
  -r REF_FILE, --ref REF_FILE
                        Ref file. Ignored when --stdin (default: None)
  -o OUT, --out OUT     output file. Default stdout (default: <_io.TextIOWrapper name='<stdout>' mode='w' encoding='utf-8'>)
  -a {skip,append,only}, --average {skip,append,only}
                        Average segment scores to produce system score. skip=do not output average (default; segment scores only);
                        append=append average at the end; only=output the average only (i.e system score only) (default: skip)
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
                        Marian backend interface. subprocess looks for marian binary in PATH. pymarian is a pybind wrapper (default:
                        pymarian)

```

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

# test

```

### `mtapi`


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

## QtDemo 
```
pymnarian-qt
```

---
## Developer 


```bash
# run this from root of project i.e., dir having pyproject.toml

# build a package
CMAKE_ARGS="..."
pip wheel -v .
```


## Known issues

1. In conda or mamba environment,  `.../miniconda3/envs/<envname>/bin/../lib/libstdc++.so.6: version 'GLIBCXX_3.4.30' not found`
    Install libstdcxx-ng

    ```bash
    conda install -c conda-forge libstdcxx-ng
    ```




