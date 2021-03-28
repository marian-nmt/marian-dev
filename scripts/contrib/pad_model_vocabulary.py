#!/usr/bin/env python3
# Pads a Marian model's vocabulary to have greater size.  The added tokens have
# zero probability.
# ./pad_model_vocabulary.py input.npz output.npz desired_vocab_size
#
# You'll also need to separately pad your vocabulary file like so:
# old=$(wc -l input.vocab |cut -d " " -f 1)
# (cat input.vocab; seq -f "<PADDING%g>" $((desired_vocab_size-old))) >output.vocab
#
# Warning: probably only works with shared vocabulary models.
import math
import numpy as np
import sys
import yaml

# Amend the vocab size in a raw ["special:model.yml"] data from a Marian npz.
# Returns the raw data to use for ["special:model.yml"]
def substitute_vocab_config(raw, new_size):
  print("Old yml: ", raw.tostring())
  raw_yaml = raw.tostring().decode("utf-8")
  #Python yaml doesn't like null bytes.
  if raw_yaml.endswith("\x00"):
    raw_yaml = raw_yaml[:-1]
  config = yaml.load(raw_yaml)
  config['dim-vocabs'] = [new_size] * len(config['dim-vocabs'])
  raw_yaml = yaml.dump(config)
  if raw_yaml.endswith("\n"):
    raw_yaml = raw_yaml[:-1]
  raw_yaml += "\x00"
  return np.array(bytearray(raw_yaml, 'utf-8'))

if len(sys.argv) != 4:
  print("Usage: " + sys.argv[0] + " input.npz output.npz desired_vocab_size")
  sys.exit(1)
  
resized_path = sys.argv[2]
new_size = int(sys.argv[3])
old_model = np.load(sys.argv[1])

new_model = dict(old_model)
old_size = len(old_model["Wemb"])
if old_size > new_size:
  sys.stderr.write("New size is smaller than original.  Cowardly refusing to clip vocab.\n")
  sys.exit(2)
print("Before: ", new_model["decoder_ff_logit_out_b"].shape, new_model["Wemb"].shape)
bias = new_model["decoder_ff_logit_out_b"]
new_model["decoder_ff_logit_out_b"] = np.pad(bias, [(0,0),(0,new_size - bias.shape[1])], mode='constant', constant_values = -math.inf)
new_model["Wemb"] = np.pad(new_model["Wemb"], [(0,new_size - bias.shape[1]), (0,0)], mode='constant', constant_values = 0)
print("After: ", new_model["decoder_ff_logit_out_b"].shape, new_model["Wemb"].shape)
new_model["special:model.yml"] = substitute_vocab_config(new_model["special:model.yml"], new_size)
print("New yml: ", new_model["special:model.yml"].tostring())
np.savez(resized_path, **new_model)
