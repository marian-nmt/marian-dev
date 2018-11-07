#! /usr/bin/python

"""
Adds or updates Marian configuration options embedded in a .npz model.
Useful for debugging
"""

import sys
import numpy as np
import yaml

def main():
  if (len(sys.argv) < 2) or ((len(sys.argv) % 2) != 0):
    usage()

  model_file = sys.argv[1]
  extra_config_dict = {}
  for i in xrange(2, len(sys.argv), 2):
    key = sys.argv[i]
    val_str = sys.argv[i+1]
    val = eval(val_str)
    extra_config_dict[key] = val

  model_dict = dict(np.load(model_file))
  yaml_conf_dict = yaml.load("".join(map(chr, model_dict['special:model.yml'][:-1])))
  yaml_conf_dict.update(extra_config_dict)
  model_dict['special:model.yml'] = np.array(map(ord, yaml.dump(yaml_conf_dict)) + [0]).astype(np.int8)
  np.savez(model_file, **model_dict)

def usage():
  print >> sys.stderr, "Usage:"
  print >> sys.stderr, sys.argv[0], "model-file [key value] [...]"
  sys.exit(-1)

if __name__ == '__main__':
  main()


