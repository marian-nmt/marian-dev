from argparse import ArgumentParser
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from scipy import stats
import math
import scipy.stats as st
import sys
from statistics import stdev 


EPS = 1e-18

def clip(tensor, range):
  return np.clip(tensor, a_min = -range, a_max = range) 

def log_b(tensor, base): 
  return  np.log(tensor + EPS) / math.log(base)


def log_quantize(tensor, bit, base, curr_max = 0):
  # find max quantization center
  max = 1.0
  if curr_max > 0:
    max = curr_max

  # scale down
  tensor = tensor / max
  
  # count the number of possible centers (divided by 2 for positive and negative side)
  centers = 2**(bit-1)
  
  # quantize
  quantized_center = np.clip(np.floor(log_b(abs(tensor * (2.0 * base)/(1.0 + base)), base)), a_min = -(centers - 1), a_max = 0)

  # revert back to float32
  quantized_tensor = np.power(base, quantized_center) * max

  # restore sign
  quantized_tensor[tensor < 0] *= -1
  
  return quantized_tensor


def compute_movement(data, curr_max, BIT, BASE):
  tmp = log_quantize(data, BIT, BASE, curr_max)
   
  basepow = (tmp / curr_max).flatten()
  dataflat = data.flatten()
  top = np.sum(basepow * dataflat)
  bottom = np.sum(basepow * basepow)
  return top / bottom


def parse_args():
  parser = ArgumentParser()
  parser.add_argument("-i", "--input", nargs='+', help=".npz file to read. Put more than 1 .npz to perform model ensembling", default="model.npz")
  parser.add_argument("-o", "--output", help="output destination", default="model.compressed.npz")
  parser.add_argument("-b", "--bit", help="quantization bit", default=4, type=int)
  parser.add_argument("--kmeans", help="Readjust scale with kmeans", default=0, type=int)
  parser.add_argument("-c", "--clip", help="clipping. set 0 to disable", default=0, type=float)

  parser.add_argument("-s", "--sparse", help="compression sparsity. Will only compress X percent of the parameters. 1 to disable", default=1, type=float)
  parser.add_argument("--base", help="base", default=2.0, type=float)
  parser.add_argument("-q", "--quiet", default=False, action="store_true")
  parser.add_argument("--skip_bias", default=False, action="store_true")  
  parser.add_argument("--max_scale", default=False, action="store_true")

  return parser.parse_args()


def print_sample(tensor_name, tensor, new_tensor):
  print(tensor_name + " | shape =  ", tensor.shape, "  : ")
  print(" before ", tensor.flat[0:6])
  print(" after  ", new_tensor.flat[0:6])
  print(" unique centers : ",len(set(tensor.flat)), " -> ",  len(set(new_tensor.flat)))
  # print(" before ", set(tensor.flat))
  # print(" after ", set(new_tensor.flat))
  print("\n")


if __name__== "__main__":
  args = parse_args()

  print("Reading models: ", args.input)
  models = []
  for model_dir in args.input:
    models.append(np.load(model_dir))
  
  # prepare and ensemble the model
  new_model = dict()
  for model in models:
    for f in model.files:
      if f not in new_model:
        new_model[f] = model[f]
      elif "special" not in f:
        new_model[f] += model[f]
  
  for f in new_model:
    if "special" in f:
      continue
    new_model[f] /= len(models)
    
  print("compressing models...")
  print("  model clipping        : ", args.clip)
  print("  log quantization bit  : ", args.bit)  
  print("  log quantization base : ", args.base)
  
  total_compressed = 0
  total_uncompressed = 0
  bias_dev = []
  full_dev = []
  for k in new_model:
    # special configurations, not a Tensor. 
    if "special" in k:
      continue
    
    # skip compressing bias
    if args.skip_bias and new_model[k].shape[0] == 1: # new_model[k].size < 10000:
      print("Skipping ",k, "( size of ", new_model[k].size, " | shape = ", new_model[k].shape, ")")
      total_uncompressed += new_model[k].size
      continue
    
    total_compressed += new_model[k].size
    tmp = new_model[k]


    # apply clipping
    if args.clip > 0:
      tmp = clip(tmp, args.clip)
     
    tmp_max = np.max(abs(tmp))

    # Apply some k-means readjustment of scale factor
    if args.kmeans > 0:
      for i in range(args.kmeans):
        tmp_max = (compute_movement(tmp, tmp_max, args.bit, args.base))
      
    tmp = log_quantize(tmp, args.bit, args.base, tmp_max)
    
    if args.sparse < 1:
      total_compressed -= np.count_nonzero(reserved)
      total_uncompressed += np.count_nonzero(reserved)
      if not args.quiet:
        print("compressing ", tmp.size - np.count_nonzero(reserved) ," / ", tmp.size)
      tmp[reserved != 0] = reserved[reserved != 0] 
    
    if not args.quiet:
      print_sample(k, new_model[k], tmp)
    new_model[k] = tmp



  print(" ===============")
  print(" compressed elements    : ", total_compressed)
  print(" uncompressed elements  : ", total_uncompressed) 
  print(" compress ratio         : ", (total_uncompressed * 32 + total_compressed * args.bit)/((total_compressed + total_uncompressed) * 32))
  print(" ===============")
  print(np.average(bias_dev), np.average(full_dev))
  print("compression done")
  print("saving to " + args.output)
  np.savez(args.output, **new_model)








