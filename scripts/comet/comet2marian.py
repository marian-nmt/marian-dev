#!/usr/bin/env python3
"""
This script converts Unbabel COMET-QE models to Marian weight file.
"""

import argparse
import logging as log
import numpy as np
import yaml
from pathlib import Path

## Uncomment to see model names supported by your installed version of unbabel-comet
# from comet.models import available_metrics
# supported_comets = [m for m in available_metrics if 'qe' in m.lower()]
supported_comets = [
    'wmt20-comet-qe-da', 'wmt20-comet-qe-da-v2', 'wmt20-comet-da',
    'wmt21-comet-qe-mqm', 'wmt21-comet-qe-da', 'wmt21-comet-da',
    'Unbabel/wmt22-comet-da', 'Unbabel/wmt22-cometkiwi-da',
    'Unbabel/XCOMET-XL', 'Unbabel/XCOMET-XXL',
    'Unbabel/wmt23-cometkiwi-da-xl', 'Unbabel/wmt23-cometkiwi-da-xxl'
]
log.basicConfig(level=log.INFO)

parser = argparse.ArgumentParser(description='Convert Unbabel COMET-QE models to Marian weight file.')
inputs = parser.add_mutually_exclusive_group(required=True)
inputs.add_argument('--roberta', '-r', help='Initialize with Roberta model', action='store_true')
inputs.add_argument('--comet', '-c', help=f'COMET model path or an ID: {", ".join(supported_comets)}')
parser.add_argument('--marian', '-m', help='Output path for Marian weight file', required=True)
parser.add_argument('-s', '--add_sigmoid', help='Add final sigmoid if not already present', action='store_true')
parser.add_argument('--spm', '-spm', type=Path, help='Save tokenizer SPM file here', required=False)
args = parser.parse_args()


def load_from_huggingface(model_id):
    log.info(f"Loading transformer model from huggingface {model_id}")
    from transformers import AutoModel, AutoTokenizer
    try:
        model = AutoModel.from_pretrained(model_id, add_pooling_layer=False)
        AutoTokenizer.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        return model.eval(), getattr(tokenizer, 'vocab_file', None)
    except:
        log.error(f"Could not resolve {model_id} from huggingface")
        raise


def load_comet_model(model_path):
    from comet import load_from_checkpoint, download_model
    from transformers import AutoTokenizer

    if not Path(model_path).exists():
        if model_path not in supported_comets:
            log.info(f"Could not find {model_path}")  # maybe it's an invalid path
        log.info(f"trying to resolve download {model_path}")
        model_path = download_model(model_path)
    log.info(f"Loading COMET model from checkpoint {model_path}")
    comet_model = load_from_checkpoint(model_path)
    comet_model.eval()

    vocab_file = None
    try:
        pretrained_model = comet_model.hparams.get('pretrained_model')
        log.info(f"comet: {model_path}; pretrained: {pretrained_model}")
        if pretrained_model:
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        vocab_file =  getattr(tokenizer, 'vocab_file', None)
    except Exception as e:
        log.warning(f'Error while locating vocab file: {e}')
        pass
    return comet_model, vocab_file

if args.roberta:
    # Load the model that Unbabel based COMET on: https://huggingface.co/microsoft/infoxlm-large
    cometModel, vocab_file = load_from_huggingface("microsoft/infoxlm-large")
else:
    cometModel, vocab_file = load_comet_model(args.comet)

if args.spm:
    vocab_file = vocab_file and Path(vocab_file)
    if vocab_file and vocab_file.exists():
        if not args.spm.parent.exists():
            raise Exception(f"Directory {args.spm.parent} does not exist")
        log.info(f"Copying {vocab_file} to {args.spm}")
        args.spm.write_bytes(vocab_file.read_bytes())
    else:
        raise Exception(f"Could not locate or save the vocab file: {vocab_file}; please remove --spm argument and try downloading the file manually")

marianModel = dict()
config = dict()

model_type = type(cometModel).__name__
print("COMET model params:", cometModel.hparams, file=sys.stderr)

# are we using the xml-roberta-xl or xml-roberta-xxl model?
isXlmXL = any(pre in cometModel.hparams.get("pretrained_model") for pre in ["xlm-roberta-xl", "xlm-roberta-xxl"])

if model_type == "RegressionMetric":
    config["type"] = "comet"
elif model_type == "ReferencelessRegression":
    config["type"] = "comet-qe"
elif model_type == "XLMRobertaModel":
    config["type"] = "comet-qe"
elif model_type == "UnifiedMetric" or isXlmXL:
    config["type"] = "comet-unified"
    config["input-join-fields"] = True
    config["separator-symbol"] = "</s>"
    config["comet-use-separator"] = True
else:
    raise Exception(f'Unknown type of model {model_type}')

config["tied-embeddings-all"] = True
config["tied-embeddings-src"] = False
config["transformer-ffn-depth"] = 2
config["transformer-ffn-activation"] = "gelu" # figure this out dynamically
config["transformer-train-position-embeddings"] = True

# Roberta-XXL (hence XCOMET-XXL) has pre-norm
if isXlmXL:
    config["transformer-preprocess"] = "n"
    config["transformer-postprocess"] = "da"
    config["transformer-postprocess-emb"] = ""
    config["transformer-postprocess-top"] = "n"
else:
    config["transformer-preprocess"] = ""
    config["transformer-postprocess"] = "dan"
    config["transformer-postprocess-emb"] = "nd"

config["bert-train-type-embeddings"] = False
config["bert-type-vocab-size"] = 0
config["comet-prepend-zero"] = True

config["comet-pool"] = cometModel.hparams.get("pool", "cls")
config["comet-mix"] = cometModel.hparams.get("layer") == "mix"
config["comet-mix-norm"] = cometModel.hparams.get('layer_norm', False)
config["comet-mix-transformation"] = cometModel.hparams.get("layer_transformation", "softmax");

# there are several issues in their code that make the following always true regardless of values in hparams
# that was hard to find out
if model_type == "UnifiedMetric" or isXlmXL:
    config["comet-mix-transformation"] = "softmax"
    config["comet-pool"] = "cls"

if not args.roberta:
    config["comet-final-sigmoid"] = args.add_sigmoid
    config["comet-pooler-ffn"] = [2048, 1024]

# @TODO: figure out if it's worth adding `cometModel.name_or_path` to the end of this version string.
config["version"] = "comet2marian2.py conversion"
config["enc-depth"] = 0

def yaml2np(config):
    configYamlStr = yaml.dump(config, default_flow_style=False)
    print("\nMarian config:")
    print(configYamlStr)

    desc = bytes(configYamlStr, 'ascii') + b'\x00'
    npDesc = np.chararray((len(desc),))
    npDesc.dtype = np.int8
    for i, b in enumerate(desc):
        npDesc[i] = b
    return npDesc

def convert(pd, srcs, trg, transpose=True, bias=False):
    # make sure exactly one element of list srcs exists in dictionary pd
    found = sum([src in pd for src in srcs])
    assert found == 1, f"Found {found} of {srcs} in {pd}"

    for src in srcs:
        if src not in pd:
            continue
        num = pd[src].detach().numpy()
        if bias:
            marianModel[trg] = num.copy()
        else:
            if transpose:
                marianModel[trg] = np.transpose(num).copy()
            else:
                marianModel[trg] = num


def match(regex, string):
    import re
    return re.search(regex, string) is not None

def extract(layer, nth, level):
    name = type(layer).__name__
    print("  " * level, nth, name)

    if match(r"Roberta(XL+)?Layer", name):
        pd = dict(layer.named_parameters())
        for n in pd:
            print("  " * (level + 1), n, pd[n].shape)

        prefix = "CometEncoder"

        blockPrefix = f"{prefix}->encoder->layers->at({nth})->as<marian::nn::TransformerEncoderLayer>()->selfAttentionBlock"

        # self-attention
        # query transformation
        convert(pd, ["attention.self.query.weight"],       f"{blockPrefix}->selfAttention->qProj->weight")
        convert(pd, ["attention.self.query.bias"],         f"{blockPrefix}->selfAttention->qProj->bias", bias=True)

        # key transformation
        convert(pd, ["attention.self.key.weight"],         f"{blockPrefix}->selfAttention->kProj->weight")
        convert(pd, ["attention.self.key.bias"],           f"{blockPrefix}->selfAttention->kProj->bias", bias=True)

        # values transformation
        convert(pd, ["attention.self.value.weight"],       f"{blockPrefix}->selfAttention->vProj->weight")
        convert(pd, ["attention.self.value.bias"],         f"{blockPrefix}->selfAttention->vProj->bias", bias=True)

        # output transformation
        convert(pd, ["attention.output.dense.weight"],     f"{blockPrefix}->selfAttention->oProj->weight")
        convert(pd, ["attention.output.dense.bias"],       f"{blockPrefix}->selfAttention->oProj->bias", bias=True)

        # self-attention layer-norm
        if isXlmXL:
            convert(pd, ["attention.self_attn_layer_norm.weight"], f"{blockPrefix}->preprocessor->norm->weight", bias=True)
            convert(pd, ["attention.self_attn_layer_norm.bias"],   f"{blockPrefix}->preprocessor->norm->bias", bias=True)
        else:
            convert(pd, ["attention.output.LayerNorm.weight"], f"{blockPrefix}->postprocessor->norm->weight", bias=True)
            convert(pd, ["attention.output.LayerNorm.bias"],   f"{blockPrefix}->postprocessor->norm->bias", bias=True)

        # ffn
        # first ffn layer
        blockPrefix = f"{prefix}->encoder->layers->at({nth})->as<marian::nn::TransformerEncoderLayer>()->filterBlock"

        convert(pd, ["intermediate.dense.weight"],         f"{blockPrefix}->layers->at(0)->as<marian::nn::Linear>()->weight")
        convert(pd, ["intermediate.dense.bias"],           f"{blockPrefix}->layers->at(0)->as<marian::nn::Linear>()->bias", bias=True)
        # second ffn layer
        convert(pd, ["output.dense.weight"],               f"{blockPrefix}->layers->at(3)->as<marian::nn::Linear>()->weight")
        convert(pd, ["output.dense.bias"],                 f"{blockPrefix}->layers->at(3)->as<marian::nn::Linear>()->bias", bias=True)

        # ffn layer-norm
        if isXlmXL:
            convert(pd, ["LayerNorm.weight"], f"{blockPrefix}->preprocessor->norm->weight", bias=True)
            convert(pd, ["LayerNorm.bias"],   f"{blockPrefix}->preprocessor->norm->bias", bias=True)
        else:
            convert(pd, ["output.LayerNorm.weight"], f"{blockPrefix}->postprocessor->norm->weight", bias=True)
            convert(pd, ["output.LayerNorm.bias"],   f"{blockPrefix}->postprocessor->norm->bias", bias=True)

        config["transformer-dim-ffn"] = pd["intermediate.dense.bias"].shape[-1]
        config["transformer-heads"] = layer.attention.self.num_attention_heads
        config["enc-depth"] += 1

    elif match(r"Roberta(XL+)?Embeddings", name):
        for n, p in layer.named_parameters():
            print("  " * (level + 1), n, p.shape)
        pd = dict(layer.named_parameters())

        # shift word embeddings so that we are back at 250,000 vocab items
        npWembTemp = pd["word_embeddings.weight"].detach().numpy()
        npWemb = npWembTemp[1:-1, :].copy()
        npWemb[0, :] = npWembTemp[0, :]
        npWemb[2, :] = npWembTemp[2, :]

        # XCOMET-XXL has some additional tokens (why?), we truncate it back to normal size
        npWemb = npWemb[0:250000, :].copy()

        marianModel["Wemb"] = npWemb

        prefix = "CometEncoder"

        # shift position embeddings so that we are back at 512 items and start at 0
        npPos = pd["position_embeddings.weight"].detach().numpy()
        npPos = npPos[2:, :].copy()
        marianModel[f"{prefix}->encoder->positionEmbedding->embeddings"] = npPos

        config["dim-emb"]    =   npWemb.shape[1]
        config["dim-vocabs"] = [ npWemb.shape[0] ]
        config["max-length"] = npPos.shape[0]

    elif match(r"Roberta(XL+)?Model", name):
        pd = dict(layer.named_parameters())
        prefix = "CometEncoder"

        # post-embedding layer normalization
        if isXlmXL:
            convert(pd, ["encoder.LayerNorm.weight"], f"{prefix}->encoder->postprocessor->norm->weight", bias=True)
            convert(pd, ["encoder.LayerNorm.bias"],   f"{prefix}->encoder->postprocessor->norm->bias", bias=True)
        else:
            convert(pd, ["embeddings.LayerNorm.weight"], f"{prefix}->encoder->preprocessor->norm->weight", bias=True)
            convert(pd, ["embeddings.LayerNorm.bias"],   f"{prefix}->encoder->preprocessor->norm->bias", bias=True)

        # on this level we actually keep recursing
        recurse(layer, level + 1)


    elif name == "LayerwiseAttention":
        for n, p in layer.named_parameters():
            print("  " * (level + 1), n, p.shape)
        pd = dict(layer.named_parameters())

        # mix layers
        weights = []
        for i in range(config["enc-depth"] + 1):
            weights.append(pd[f"scalar_parameters.{i}"].detach().numpy())
        marianModel["CometEncoder->encoder->weights"] = np.concatenate(weights).copy()

        # gamma for weird batch/layer-norm step in pooler/encoder of COMET
        # @TODO: make optional
        marianModel["CometEncoder->encoder->gamma"] = pd["gamma"].detach().numpy().copy()

    elif name == "FeedForward":
        for n, p in layer.named_parameters():
            print("  " * (level + 1), n, p.shape)
        pd = dict(layer.named_parameters())

        if layer.ff[-1].__class__.__name__ == "Sigmoid" or args.add_sigmoid:
            config["comet-final-sigmoid"] = True

        config["comet-pooler-ffn"] = [
            pd["ff.0.bias"].shape[0],
            pd["ff.3.bias"].shape[0]
        ]

        # 3-layer FFN network that computes COMET regression
        prefix = "CometQEPooler"

        convert(pd, ["ff.0.weight"], f"{prefix}->layers->at(0)->as<marian::nn::Linear>()->weight")
        convert(pd, ["ff.0.bias"],   f"{prefix}->layers->at(0)->as<marian::nn::Linear>()->bias", bias=True)

        convert(pd, ["ff.3.weight"], f"{prefix}->layers->at(3)->as<marian::nn::Linear>()->weight")
        convert(pd, ["ff.3.bias"],   f"{prefix}->layers->at(3)->as<marian::nn::Linear>()->bias", bias=True)

        convert(pd, ["ff.6.weight"], f"{prefix}->layers->at(6)->as<marian::nn::Linear>()->weight")
        convert(pd, ["ff.6.bias"],   f"{prefix}->layers->at(6)->as<marian::nn::Linear>()->bias", bias=True)
    else:
        recurse(layer, level + 1)

def recurse(parent, level=0):
    for i, child in enumerate(parent.children()):
        extract(child, i, level)

recurse(cometModel)
marianModel["special:model.yml"] = yaml2np(config)

for m in marianModel:
    print(m, marianModel[m].shape)

print("Saving Marian model to %s" % (args.marian,))
np.savez(args.marian, **marianModel)
