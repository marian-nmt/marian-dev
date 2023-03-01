#!/usr/bin/env python3
"""
This script converts Unbabel COMET-QE models to Marian weight file.
"""

import argparse
import yaml
import numpy as np

parser = argparse.ArgumentParser(description='Convert Unbabel COMET-QE models to Marian weight file.')
inputs = parser.add_mutually_exclusive_group(required=True)
inputs.add_argument('--comet', help='Path to COMET model')
inputs.add_argument('--roberta', help='Initialize with Roberta model', action='store_true')
parser.add_argument('--marian', help='Output path for Marian weight file', required=True)
parser.add_argument('-s', '--add_sigmoid', help='Add final sigmoid if not already present', action='store_true')
args = parser.parse_args()


if args.roberta:
    from transformers import AutoModel
    # Load the model that Unbabel based COMET on: https://huggingface.co/microsoft/infoxlm-large
    robertaModel = AutoModel.from_pretrained("microsoft/infoxlm-large", add_pooling_layer=False)
    robertaModel.eval()
    print(robertaModel)
    cometModel = robertaModel
else:
    from comet import load_from_checkpoint
    cometModel = load_from_checkpoint(args.comet)
    cometModel.eval()
    print(cometModel)

marianModel = dict()

config = dict()
config["type"] = "comet-qe"
config["tied-embeddings-all"] = True
config["tied-embeddings-src"] = False
config["transformer-ffn-depth"] = 2
config["transformer-ffn-activation"] = "gelu" # figure this out dynamically
config["transformer-train-position-embeddings"] = True
config["transformer-preprocess"] = ""
config["transformer-postprocess"] = "dan"
config["transformer-postprocess-emb"] = "nd"
config["bert-train-type-embeddings"] = False
config["bert-type-vocab-size"] = 0
config["comet-prepend-zero"] = True
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
    if len(srcs) == 1:
        for src in srcs:
            num = pd[src].detach().numpy()
            if bias:
                marianModel[trg] = num.copy()
            else:
                if transpose:
                    marianModel[trg] = np.transpose(num).copy()
                else:
                    marianModel[trg] = num
    else: # path that joins matrices together for fused self-attention
        nums = [pd[src].detach().numpy() for src in srcs]
        if bias:
            nums = [np.transpose(num) for num in nums]
        marianModel[trg] = np.stack(nums, axis=0).copy()

def extract(layer, nth, level):
    name = type(layer).__name__
    print("  " * level, nth, name)
    if "RobertaLayer" in name:
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
        convert(pd, ["output.LayerNorm.weight"],           f"{blockPrefix}->postprocessor->norm->weight", bias=True)
        convert(pd, ["output.LayerNorm.bias"],             f"{blockPrefix}->postprocessor->norm->bias", bias=True)

        config["transformer-dim-ffn"] = pd["intermediate.dense.bias"].shape[-1]
        config["transformer-heads"] = layer.attention.self.num_attention_heads
        config["enc-depth"] += 1

    elif "RobertaEmbeddings" in name:
        for n, p in layer.named_parameters():
            print("  " * (level + 1), n, p.shape)
        pd = dict(layer.named_parameters())

        # shift word embeddings so that we are back at 250,000 vocab items
        npWembTemp = pd["word_embeddings.weight"].detach().numpy()
        npWemb = npWembTemp[1:-1, :].copy()
        npWemb[0, :] = npWembTemp[0, :]
        npWemb[2, :] = npWembTemp[2, :]
        marianModel["Wemb"] = npWemb

        prefix = "CometEncoder"
        
        # shift position embeddings so that we are back at 512 items and start at 0
        npPos = pd["position_embeddings.weight"].detach().numpy()
        npPos = npPos[2:, :].copy()
        marianModel[f"{prefix}->encoder->positionEmbedding->embeddings"] = npPos

        # post-embedding layer normalization
        convert(pd, ["LayerNorm.weight"], f"{prefix}->encoder->preprocessor->norm->weight", bias=True)
        convert(pd, ["LayerNorm.bias"],   f"{prefix}->encoder->preprocessor->norm->bias", bias=True)

        config["dim-emb"]    =   npWemb.shape[1]
        config["dim-vocabs"] = [ npWemb.shape[0] ]
        config["max-length"] = npPos.shape[0]

    elif name == "LayerwiseAttention":
        for n, p in layer.named_parameters():
            print("  " * (level + 1), n, p.shape)
        pd = dict(layer.named_parameters())

        # mix layers
        weights = []
        for i in range(25):
            weights.append(pd[f"scalar_parameters.{i}"].detach().numpy())
        marianModel["CometEncoder->encoder->weights"] = np.concatenate(weights).copy()

        # gamma for weird batch/layer-norm step in pooler/encoder of COMET
        # @TODO: make optional
        marianModel["CometEncoder->encoder->gamma"] = pd["gamma"].detach().numpy().copy()
        config["comet-mix"] = True
        config["comet-mix-norm"] = True
        

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

        # @TODO: make final sigmoid optional
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
