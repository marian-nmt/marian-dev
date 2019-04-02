from argparse import ArgumentParser
import numpy as np
import math


def clip(tensor, range):
  return np.clip(tensor, a_min = -range, a_max = range) 


def ratio_clip(tensor, rate):
  s_tensor = abs(tensor).flatten()
  s_tensor.sort()
  range = s_tensor[int(rate * tensor.size)]
  print("clip ", range)
  return clip(tensor, range)


def log_b(tensor, base):
  return  np.log(tensor) / math.log(base)


def log_quantize(tensor, bit, base):
  # find max quantization center
  max = np.max(np.abs(tensor))
  max = base ** math.floor(math.log(max) / math.log(base))

  # scale down
  tensor = tensor / max
  
  # count the number of possible centers (divided by 2 for positive and negative side)
  centers = 2**(bit-1)
  
  # quantize
  quantized_center = np.clip(np.round(log_b(abs(tensor), base)), a_min = -(centers - 1), a_max = 0)
  
  # set positive C^0 to C^1
  # quantized_center[(quantized_center >= 0) & (tensor > 0)] = -1  

  # revert back to float32
  quantized_tensor = np.power(base, quantized_center) * max

  # restore sign
  quantized_tensor[tensor < 0] *= -1
  
  # set some to zero
  # quantized_tensor[quantized_center <= -centers] = 0

  return quantized_tensor



def parse_args():
  parser = ArgumentParser()
  parser.add_argument("-i", "--input", help=".npz file to read", default="model.npz")
  parser.add_argument("-o", "--output", help="output destination", default="model.compressed.npz")
  parser.add_argument("-b", "--bit", help="quantization bit", default=4, type=int)
  parser.add_argument("-c", "--clip", help="clipping. set 0 to disable", default=0, type=int)
  parser.add_argument("-q", "--quiet", default=False, action="store_true")
  return parser.parse_args()


def print_sample(tensor_name, tensor, new_tensor):
  print(tensor_name + " : ")
  print(" before ", tensor.flat[0:6])
  print(" after  ", new_tensor.flat[0:6])
  print(" unique centers : ",len(set(tensor.flat)), " -> ",  len(set(new_tensor.flat)))
  print("\n")


if __name__== "__main__":

  args = parse_args()

  print("Reading " + args.input)
  
  model = np.load(args.input)
  new_model = dict((f, model[f]) for f in model.files)
  

  print("compressing models...")
  print("  model clipping       : ", args.clip)
  print("  log quantization bit : ", args.bit)

  for k in new_model:
    # special configurations, not a Tensor. 
    if "special" in k:
      continue
    tmp = new_model[k]

    if args.clip > 0:
      tmp = clip(tmp, args.clip)
    
    tmp = log_quantize(tmp, args.bit, 2.0)
    # tmp = log_compress(tmp, 1.0, args.bit, 2.0)
    if not args.quiet:
      print_sample(k, new_model[k], tmp)
    new_model[k] = tmp


  print("compression done")
  print("saving to " + args.output)
  np.savez(args.output, **new_model)








