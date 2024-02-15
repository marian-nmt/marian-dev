#!/usr/bin/env bash
set -eu
MYDIR=$(dirname ${BASH_SOURCE[0]})
cd $MYDIR

ENV_NAME=metrics
which conda > /dev/null || (echo "conda not found" && exit 1)
# conda functions are not exported in non-interactive shell, so we source conda.sh
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
FOUND="$(conda env list  | awk -v name=$ENV_NAME '$1==name { print $1 }')"

log() {
    echo -e "\e[32m$@\e[0m" >&2
}
#### SETUP #########
if [[ -z "$FOUND" ]]; then
    log "Creating conda environment $ENV_NAME"
    # create conda environment and install requirements
    conda create -n $ENV_NAME python=3.10
    conda activate $ENV_NAME
    log "Installing requirements"
    pip install -r $MYDIR/requirements.txt
else
    log "Activating conda environment $ENV_NAME"
    conda activate $ENV_NAME
fi

which pymarian-eval > /dev/null || (
    echo "pymarian-eval not found. Please install and return" && exit 1 )

#####################
bash ./compare.sh