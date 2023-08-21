# Marian Evaluate
The main script is `compare.sh`, however it needs to be run in an environment where all three -- marian, unbabel-comet(pytorch), and bleurt(tensorflow) are available.
Hence, 1) we create a docker container with all the necessary libs.
    and 2) run compare.sh inside the docker environment

## Setup: build docker image

```bash
./setup.sh
```

## Run compare.sh in docker container

```bash
./docker-run.sh
```
The `docker-run.sh` script mounts cache directory from the host to container.
The necessary files (weights and vocabularies) will be automatically downloaded and cached for unbabel-comet and Bleurt metrics.
However, for `marian-score.sh` expects the cache to be prepared under `$HOME/.cache/marian/metrics`.
The structure/format of the cache directory for marian-score.sh looks as follows:
```bash
/home/$USER/.cache/marian/metrics/
├── bleurt20-ref
│   ├── bleurt-20.model.npz
│   ├── bleurt.vocab.spm
├── comet20-da-src
│   ├── comet20-qe-da.model.npz
│   └── roberta.vocab.spm
└── comet20-da-src+ref
    ├── comet20-da.model.npz
    └── roberta.vocab.spm
```
Each metric subdir should have a `*model.npz` and a `*vocab.spm` files, and the name of metric directory should end with `-src|-qe|-ref|-src+ref` suffix to indicate the category of metric.

> TODO: Upload Marian compatible comet and bleurt models to public blob storage and modify script to automatically download

