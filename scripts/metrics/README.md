# Marian Metrics

The main script is `compare.sh`, however it needs to be run in an environment where all three -- marian, unbabel-comet(pytorch), and bleurt(tensorflow) are available.
Hence we create a new python environment using conda to run comparisons.

## Setup

```bash
./run.sh
```
This setups a conda environment named `metrics` which will have all the necessary requirements, except pymarian-eval, which you will have to install based on your CMAKE settings
```bash
# from the root dir of this repository
conda activate metrics
mkdir build; cd build
cmake .. -DPYMARIAN=on #.. other flags
pip install pymarian-*.whl
```

## Run Compare.sh

```bash

# option 1:
./run.sh

# option 2
conda activate metrics
bash compare.sh
```

This script produces reports at  `workspace/*.report.txt`, which shows average difference segment level scores between original implementation and `pymarian-eval`

## Convert Metrics Weights to Marian format

```bash
conda activate metrics
MARIAN=../build/marian ./convert-all-models.sh
```

To add a new model ID, edit `known-models.txt` file in the same directory as this README
