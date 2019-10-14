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


def ratio_clip(tensor, rate):
  s_tensor = abs(tensor).flatten()
  s_tensor.sort()
  range = s_tensor[int(rate * tensor.size)]
  print("clip ", range)
  return clip(tensor, range)


def log_b(tensor, base): 
  return  np.log(tensor + EPS) / math.log(base)


def log_quantize(tensor, bit, base, curr_max = 0):
  # find max quantization center
  # max = np.max(np.abs(tensor))
  # max = base ** math.floor(math.log(max * 4.0 / 3.0) / math.log(base))
  max = 1.0
  if curr_max > 0:
    max = curr_max

  # scale down
  tensor = tensor / max
  
  # count the number of possible centers (divided by 2 for positive and negative side)
  centers = 2**(bit-1)
  
  # quantize
  quantized_center = np.clip(np.floor(log_b(abs(tensor * (2.0 * base)/(1.0 + base)), base)), a_min = -(centers - 1), a_max = 0)

  # set positive C^0 to C^1
  # quantized_center[(quantized_center >= 0) & (tensor > 0)] = -1  

  # revert back to float32
  quantized_tensor = np.power(base, quantized_center) * max

  # restore sign
  quantized_tensor[tensor < 0] *= -1
  
  # set some to zero
  # quantized_tensor[quantized_center <= -centers] = 0

  return quantized_tensor


def fixed_quantize(tensor, bit, max = 0):
  centers = 2**(bit-1)
  # max = 0
  # find max quantization center
  if max == 0:
    maks = np.max(tensor)
    mins = np.min(tensor)
    max = np.max((maks * centers / (centers - 1), -mins))
  # max = 1
  # return tensor
  quantized_tensor = np.clip(np.round((centers * tensor) / max), a_min = -centers, a_max = (centers - 1))
  
  return quantized_tensor * max / centers

# obtain top X percent largest (by absolute) parameters.
def get_sparse(tensor, sparsity):
  index = max(0, int(tensor.size * sparsity) - 1)
  threshold = np.partition(abs(tensor).flatten(), index )[index]
  print(threshold)
  masked = np.copy(tensor)
  masked[abs(masked) < threshold] = 0

  return masked



def compute_movement(data, curr_max, BIT, BASE):
  if not args.fixed:
    tmp = log_quantize(data, BIT, BASE, curr_max)
  else:
    tmp = fixed_quantize(data, BIT, curr_max)

  print(" SME ", mean_squared_error(data, tmp))
   
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
  parser.add_argument("-f", "--fixed", default=False, action="store_true")
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


# real independent kmeans
def kmeans_assign(tensor, C):
  tmp = tensor.flat
  assigns = np.zeros(tmp.size)
  best_centroid = 0
  closest = 999999999.0


def kmean_quantize(tensor, bit, steps):
  # get abs value
  tmp = abs(tensor)
  # get starting position based log based quantization, max scale
  centers = 2**(bit-1)
  C = np.zeros(centers)
  mx = np.max(abs(tensor))
  drops = mx / centers
  for i in range(centers):
    C[i] = mx
    mx /= 2
  
  print("initial ", C)
 
  # kmeans:
  for i in range(steps):
    # assigning to class
    assigns = np.zeros(tensor.shape)
    quant = np.full(tensor.shape, C[0]).astype('float32')

    for j in range(centers):
      mask = abs(tmp - C[j]) < abs(tmp - quant)

      assigns[mask] = j
      quant[mask] = C[j]
    

    # update centroids:
    for j in range(centers):
      if np.count_nonzero(assigns == j) > 0:
        C[j] = np.average(tmp[assigns == j])
      else:
        C[j] = 0

    print("updated ", C)
    print(" SME ", mean_squared_error(tmp, quant))
  print("final ", C)
  quantized_tensor = quant
  quantized_tensor[tensor < 0] *= -1
  print(" SME ", mean_squared_error(tensor, quantized_tensor))
  return quantized_tensor 

if __name__== "__main__":
  args = parse_args()
  A = np.random.rand(2,4) - 0.5
  print(A)
  B = kmean_quantize(A, 4, 1)
  A = np.random.rand(2,4) - 0.5
  B = kmean_quantize(A, 4, 1)
  print(B)
  C = log_quantize(A, 4, 2, np.max(abs(A)))
  print(C)
  
  print("diff ", mean_squared_error(B, C))


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
  first = 4
  total_compressed = 0
  total_uncompressed = 0
  bias_dev = []
  full_dev = []
  for k in new_model:
    # if "encoder_l1_self_Wv" not in k:
    #  continue
    # special configurations, not a Tensor. 
    if "special" in k:
      continue
  
    # normality test
    k2, p = stats.normaltest(np.random.choice(new_model[k].flat, 100))
    if p < 0.05:
      print(" REJECT NULL HYPOTHESIS ", k, " : ", p)
    else:
      print(" ACCEPT NULL HYPOTHESIS ", k, " : ", p)
    
    if new_model[k].shape[0] == 1:
      bias_dev.append(np.std(new_model[k]))
    else:
      full_dev.append(np.std(new_model[k]))

    # skip compressing bias
    if args.skip_bias and new_model[k].shape[0] == 1: # new_model[k].size < 10000:
      print("Skipping ",k, "( size of ", new_model[k].size, " | shape = ", new_model[k].shape, ")")
      total_uncompressed += new_model[k].size
      continue
    
    total_compressed += new_model[k].size
    tmp = new_model[k]

    # np.savetxt(sys.stdout, np.histogram(tmp, 100)[0])

    # apply clipping
    if args.clip > 0:
      tmp = clip(tmp, args.clip)
    first -= 1
    
    # independent k-means
    if args.kmeans < 0:
      tmp = kmean_quantize(tmp, args.bit, -args.kmeans)
     
    else:
      # Adjust scale based on the data's mean
      if not args.max_scale:
        # centers_avg = 2 ** (-args.bit + 1)
        centers_avg = 0.249023438
        data_avg = np.average(abs(tmp))
        tmp_max = data_avg / centers_avg
      # Adjust scale basied on the max value
      else:
        tmp_max = np.max(abs(tmp))
        # tmp_max =  1
        # tmp_max = 2.58 * np.std(np.concatenate((tmp.flat , (-1 * tmp).flat )))

      # Apply some k-means readjustment of scale factor
      if args.kmeans > 0:
        for i in range(args.kmeans):
          print("  tmp comp center    ", tmp_max)
          tmp_max = (compute_movement(tmp, tmp_max, args.bit, args.base))
        # if (tmp_max > 1.5 * np.max(abs(tmp))):
        #  tmp_max /= 2.0 
      if not args.quiet:
        print("compression center ", tmp_max)
        print("data center        ", np.average(abs(tmp)))
      
      if not args.fixed:
        tmp = log_quantize(tmp, args.bit, args.base, tmp_max)
      else:
        tmp = fixed_quantize(tmp, args.bit, tmp_max)
    
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








