#!/usr/bin/env python3
"""
This script converts Fairseq Roberta model to Marian weight file.
"""

import argparse
import numpy as np
import sys
import torch
import yaml

from fairseq.models.roberta import RobertaModel

parser = argparse.ArgumentParser(description='Convert Fairseq Roberta model to Marian weight file.')
parser.add_argument('--roberta', help='Path to Roberta model', required=True)
parser.add_argument('--comet', help='Path to COMET model', required=True)
parser.add_argument('--marian', help='Output path for Marian weight file', required=True)
args = parser.parse_args()

roberta = RobertaModel.from_pretrained(args.roberta)
model = torch.load(args.comet)
print(model)

roberta.eval()

config = dict()
config["type"] = "bert-encoder"
config["input-types"] = ["sequence"]
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
# @TODO: figure out if it's worth adding `cometModel.name_or_path` to the end of this version string.
config["version"] = "roberta2marian.py conversion"

config["enc-depth"] = 0

marianModel = dict()

def convert(pd, srcs, trg, transpose=True, bias=False):
    if len(srcs) == 1:
        for src in srcs:
            num = pd[src].detach().numpy()
            if bias:
                marianModel[trg] = np.atleast_2d(num).copy()
            else:
                if transpose:
                    marianModel[trg] = np.transpose(num).copy()
                else:
                    marianModel[trg] = num
    else: # path that joins matrices together for fused self-attention
        nums = [pd[src].detach().numpy() for src in srcs]
        if bias:
            nums = [np.transpose(np.atleast_2d(num)) for num in nums]
        marianModel[trg] = np.stack(nums, axis=0).copy()


def extract(layer, nth, level):
    name = type(layer).__name__
    print("  " * level, nth, name)
    if name == "TransformerSentenceEncoderLayer":
        pd = dict(layer.named_parameters())
        for n in pd:
            print("  " * (level + 1), n, pd[n].shape)
    
        convert(pd, ["self_attn.q_proj.weight"], f"encoder_l{nth + 1}_self_Wq")
        convert(pd, ["self_attn.k_proj.weight"], f"encoder_l{nth + 1}_self_Wk")
        convert(pd, ["self_attn.v_proj.weight"], f"encoder_l{nth + 1}_self_Wv")

        convert(pd, ["self_attn.q_proj.bias"],   f"encoder_l{nth + 1}_self_bq", bias=True)
        convert(pd, ["self_attn.k_proj.bias"],   f"encoder_l{nth + 1}_self_bk", bias=True)
        convert(pd, ["self_attn.v_proj.bias"],   f"encoder_l{nth + 1}_self_bv", bias=True)

        # convert(pd, ["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"], f"encoder_l{nth + 1}_self_Wt")
        # convert(pd, ["self_attn.q_proj.bias",   "self_attn.k_proj.bias",   "self_attn.v_proj.bias"],   f"encoder_l{nth + 1}_self_bt", bias=True)

        convert(pd, ["self_attn.out_proj.weight"], f"encoder_l{nth + 1}_self_Wo")
        convert(pd, ["self_attn.out_proj.bias"],   f"encoder_l{nth + 1}_self_bo", bias=True)

        convert(pd, ["self_attn_layer_norm.weight"], f"encoder_l{nth + 1}_self_Wo_ln_scale", bias=True)
        convert(pd, ["self_attn_layer_norm.bias"],   f"encoder_l{nth + 1}_self_Wo_ln_bias", bias=True)

        convert(pd, ["fc1.weight"], f"encoder_l{nth + 1}_ffn_W1")
        convert(pd, ["fc1.bias"],   f"encoder_l{nth + 1}_ffn_b1", bias=True)
        convert(pd, ["fc2.weight"], f"encoder_l{nth + 1}_ffn_W2")
        convert(pd, ["fc2.bias"],   f"encoder_l{nth + 1}_ffn_b2", bias=True)

        convert(pd, ["final_layer_norm.weight"], f"encoder_l{nth + 1}_ffn_ffn_ln_scale", bias=True)
        convert(pd, ["final_layer_norm.bias"],   f"encoder_l{nth + 1}_ffn_ffn_ln_bias", bias=True)

        config["transformer-dim-ffn"] = pd["fc1.bias"].shape[-1]
        config["transformer-heads"] = layer.self_attn.num_heads
        config["enc-depth"] += 1

    elif name == "Embedding":
        for n, p in layer.named_parameters():
            print("  " * (level + 1), n, p.shape)
        pd = dict(layer.named_parameters())
        convert(pd, ["weight"], f"Wemb", transpose=False)

        config["dim-emb"] = pd["weight"].shape[1]
        config["dim-vocabs"] = [ pd["weight"].shape[0] ]

    elif name == "LearnedPositionalEmbedding":
        for n, p in layer.named_parameters():
            print("  " * (level + 1), n, p.shape)
        pd = dict(layer.named_parameters())
        convert(pd, ["weight"], f"Wpos", transpose=False)

        config["max-length"] = pd["weight"].shape[0]

    elif name == "RobertaLMHead":
        for n, p in layer.named_parameters():
            print("  " * (level + 1), n, p.shape)
    
        pd = dict(layer.named_parameters())
        convert(pd, ["dense.weight"],      f"masked-lm_ff_logit_l1_W")
        convert(pd, ["dense.bias"],        f"masked-lm_ff_logit_l1_b", bias=True)
        convert(pd, ["layer_norm.weight"], f"masked-lm_ff_ln_scale", bias=True)
        convert(pd, ["layer_norm.bias"],   f"masked-lm_ff_ln_bias", bias=True)
        
        convert(pd, ["bias"],              f"masked-lm_ff_logit_l2_b", bias=True)
        # reuse Wemb here as weight
        # convert(pd, ["weight"],    f"masked-lm_ff_logit_l2_b")
        
    elif name == "LayerNorm":
        for n, p in layer.named_parameters():
            print("  " * (level + 1), n, p.shape)

        pd = dict(layer.named_parameters())
        convert(pd, ["weight"], f"encoder_emb_ln_scale_pre", bias=True)
        convert(pd, ["bias"],   f"encoder_emb_ln_bias_pre", bias=True)

    else:
        recurse(layer, level + 1)

def recurse(parent, level=0):
    for i, child in enumerate(parent.children()):
        extract(child, i, level)
        
recurse(roberta)

for m in marianModel:
    print(m, marianModel[m].shape)

configYamlStr = yaml.dump(config, default_flow_style=False)
desc = list(configYamlStr)
npDesc = np.chararray((len(desc),))
npDesc[:] = desc
npDesc.dtype = np.int8
marianModel["special:model.yml"] = npDesc

print("\nMarian config:")
print(configYamlStr)
print("Saving Marian model to %s" % (args.marian,))
np.savez(args.marian, **marianModel)