#!/usr/bin/env bash
set -eu
MYDIR=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)
SCRIPTS=$(cd $MYDIR/.. && pwd)

: "
This script converts all metrics models to Marian format (.npz) and converts them to memory maps (.bin).
This script expects comet2marian.py and bleurt2marian.py
The script also expects marian binary to be in PATH or set as MARIAN environment variable.

Pre-requisites:
    pip install unbabel-comet
Optionally, you may need to configure huggingface transformers,
 specifically, hf-login for models that reqire login (e.g., wmt22-cometkiwi-da).

To run bleurt2marian, install bleurt-pytorch package:
    pip install git+https://github.com/lucadiliello/bleurt-pytorch.git
"

OUT_DIR=${1:-$MYDIR/marian-metric}   # NOTE: manually copy this to  /mnt/textmt/www/marian/metric
COMET2MARIAN=$SCRIPTS/comet/comet2marian.py
BLEURT2MARIAN=$SCRIPTS/bleurt/bleurt2marian.py
MARIAN=${MARIAN:-}

# locate marian binary
if [[ -z "$MARIAN" ]]; then
    if [[ -f $SCRIPTS/../build/marian ]]; then
        MARIAN=$SCRIPTS/../build/marian
    elif which marian > /dev/null; then
        MARIAN=$(which marian)
    fi
fi
if [[ -z "$MARIAN" || ! -e $MARIAN ]]; then
    echo -e "Error: marian binary not found." \
        "\n  Option 1) export MARIAN=path/to/marian" \
        "\n  Option 2) make sudo <repository>/build/marian exists" \
        "\n  Option 2) add marian binary to PATH" >&2
    exit 1
fi

if [[ ! -f $COMET2MARIAN ]]; then
    echo "comet2marian.py not found at $COMET2MARIAN"; exit 2
fi
if [[ ! -f $BLEURT2MARIAN ]]; then
    echo "bleurt2marian.py not found at $BLEURT2MARIAN"; exit 2
fi

MODEL_IDS=$(cat $MYDIR/known-models.txt | grep -v '^#' | awk '{print $1}')


######## convert to marian #########
for model_id in ${MODEL_IDS[@]}; do
    # lowercase model name
    model_name=$(basename $model_id | tr '[:upper:]' '[:lower:]')
    model_dir=$OUT_DIR/$model_name
    ok_flag=$model_dir/._OK
    if [[ -f $ok_flag ]]; then
        echo "$model_id already exists at $model_dir, skipping." >&2
        continue
    fi
    echo "Creating $model_dir"
    mkdir -p $model_dir
    npz_file=$model_dir/model.$model_name.npz
    bin_file=${npz_file%.npz}.bin

    # step 1 create .npz file
    if [[ ! -f $npz_file || ! -f $npz_file.md5 ]]; then
        CONVERT=""
        if [[ $model_id =~ BLEURT ]]; then
            # only one BLEURT model supported, so it does not take model ID
            CONVERT="$BLEURT2MARIAN"
        else
            CONVERT="$COMET2MARIAN -c $model_id"
        fi
        rm -f $npz_file $npz_file.md5 # remove incomplete files
        ${CONVERT} -m $npz_file --spm $model_dir/vocab.spm \
            || { echo "Error: failed to convert $model_id to Marian format" >&2; exit 3; }
        md5sum $npz_file | awk '{print $1}' > $npz_file.md5
    fi

    # Step 2: convert to memory map
    if [[ ! -f $bin_file || ! -f $bin_file.md5  ]]; then
        echo "Convert $npz_file --> $bin_file"
        rm -f $bin_file $bin_file.md5  # remove incomplete files
        $MARIAN convert -f $npz_file -t $bin_file || {
            echo "Error: failed to convert $npz_file to memory map" >&2; exit 4;
        }
        md5sum $bin_file | awk '{print $1}' > $bin_file.md5
    fi
    touch $ok_flag
done

# NOTE: only update the new/changed models
#cp -r $OUT_DIR/* /mnt/textmt/www/marian/metric