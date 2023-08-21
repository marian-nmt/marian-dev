#!/usr/bin/env python3
"""
This script converts Google BLEURT models to Marian weight file.
"""

import argparse
import logging as log
import numpy as np
import yaml
from pathlib import Path

BLEURT_LOCATION = 'lucadiliello/BLEURT-20'

log.basicConfig(level=log.INFO)

parser = argparse.ArgumentParser(description='Convert Google BLEURT models to Marian weight file.')
parser.add_argument('--marian', '-m', help='Output path for Marian weight file', required=True)
parser.add_argument('--spm', '-spm', type=Path, help='Save tokenizer SPM file here', required=False)
args = parser.parse_args()

def load_bleurt_model():
    from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer

    bleurt_model = BleurtForSequenceClassification.from_pretrained(BLEURT_LOCATION)
    bleurt_model.eval()
    tokenizer = BleurtTokenizer.from_pretrained(BLEURT_LOCATION)
    vocab_file = None
    if tokenizer.vocab_file and Path(tokenizer.vocab_file).exists():
        vocab_file = tokenizer.vocab_file
    return bleurt_model, vocab_file

bleurt_model, vocab_file = load_bleurt_model()

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

config["type"] = "bleurt"
config["tied-embeddings-all"] = True
config["tied-embeddings-src"] = False
config["transformer-ffn-depth"] = 2
config["transformer-ffn-activation"] = "gelu" # figure this out dynamically
config["transformer-train-position-embeddings"] = True
config["transformer-preprocess"] = ""
config["transformer-postprocess"] = "dan"
config["transformer-postprocess-emb"] = "nd"
config["bert-train-type-embeddings"] = True
config["bert-type-vocab-size"] = 2
config["comet-prepend-zero"] = True
config["input-join-fields"] = True
config["version"] = "bleurt2marian.py conversion"
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

    if "BleurtEncoder" in name:
        # embedding projection
        prefix = "BleurtEncoder"

        pd = dict(layer.named_parameters())
        for n in pd:
            if "embedding_projection" in n:
                print("  " * (level + 1), n, pd[n].shape)

        convert(pd, ["embedding_projection.weight"], f"{prefix}->encoder->eProj->weight")
        convert(pd, ["embedding_projection.bias"],   f"{prefix}->encoder->eProj->bias", bias=True)
        
        # continue recursing down the model structure
        recurse(layer, level + 1)

    elif "BleurtLayer" in name:
        pd = dict(layer.named_parameters())
        for n in pd:
            print("  " * (level + 1), n, pd[n].shape)

        prefix = "BleurtEncoder"
        blockPrefix = f"{prefix}->encoder->layers->at({nth})->as<marian::nn::TransformerEncoderLayer>()->selfAttentionBlock"

        if not "transformer-dim-model" in config:
            query = pd["attention.self.query.weight"].detach().numpy()
            config["transformer-dim-model"] = query.shape[1]

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

    elif "BleurtEmbeddings" in name:
        for n, p in layer.named_parameters():
            print("  " * (level + 1), n, p.shape)
        pd = dict(layer.named_parameters())

        # @TODO: this is a dirty trickery and should be solved differently in the future
        npWemb = pd["word_embeddings.weight"].detach().numpy()
        # put embedding of [CLS] in place of [PAD] (0)
        npWemb[0, :] = npWemb[312, :]
        # put embedding of [SEP] in place of </s>
        npWemb[1, :] = npWemb[313, :]
        marianModel["Wemb"] = npWemb

        prefix = "BleurtEncoder"
        
        npPos = pd["position_embeddings.weight"].detach().numpy()
        # this should be moved out of the encoder into a special embedding layer
        marianModel[f"{prefix}->encoder->positionEmbedding->embeddings"] = npPos
        
        npType = pd["token_type_embeddings.weight"].detach().numpy()
        marianModel[f"{prefix}->typeEmbedding->embeddings"] = npType

        # post-embedding layer normalization
        convert(pd, ["LayerNorm.weight"], f"{prefix}->encoder->preprocessor->norm->weight", bias=True)
        convert(pd, ["LayerNorm.bias"],   f"{prefix}->encoder->preprocessor->norm->bias", bias=True)

        config["dim-emb"]    =   npWemb.shape[1]
        config["dim-vocabs"] = [ npWemb.shape[0] ]
        config["max-length"] = npPos.shape[0]        

    # this will be the bleurt pooler right here:
    elif name == "BleurtPooler":
        for n, p in layer.named_parameters():
            print("  " * (level + 1), n, p.shape)
        pd = dict(layer.named_parameters())


        prefix = "BleurtPooler"
        convert(pd, ["dense.weight"], f"{prefix}->layers->at(0)->as<marian::nn::Linear>()->weight")
        convert(pd, ["dense.bias"],   f"{prefix}->layers->at(0)->as<marian::nn::Linear>()->bias", bias=True)

    else:
        recurse(layer, level + 1)

def recurse(parent, level=0):
    for i, child in enumerate(parent.children()):
        extract(child, i, level)

recurse(bleurt_model)

# last layer
prefix = "BleurtPooler"
pd = dict(bleurt_model.named_parameters())
convert(pd, ["classifier.weight"], f"{prefix}->layers->at(3)->as<marian::nn::Linear>()->weight")
convert(pd, ["classifier.bias"],   f"{prefix}->layers->at(3)->as<marian::nn::Linear>()->bias", bias=True)

marianModel["special:model.yml"] = yaml2np(config)

for m in marianModel:
    print(m, marianModel[m].shape)

print("Saving Marian model to %s" % (args.marian,))
np.savez(args.marian, **marianModel)