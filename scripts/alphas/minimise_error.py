#!/usr/bin/env python3
import sys
import numpy as np

def get_quant_error(matrix, quantNum):
    quantmult = np.float32(127)/np.float32(quantNum)

    #Quantize
    quantized_matrix = (matrix*quantmult).astype(np.int8)

    #Unquantize
    unquantized_matrix = quantized_matrix*(1/quantmult)

    return np.square(np.subtract(unquantized_matrix, matrix)).mean()

def find_best_matrix(matrix):
    maxAbs = np.float32(abs(max(matrix.min(), matrix.max(), key=abs)))
    maxAbs = maxAbs + np.float32(0.0001) # The first condition
    bestFactor = maxAbs
    bestMSE =  np.inf
    increases = 0

    while maxAbs > 0 and increases < 20:
        maxAbs = maxAbs - np.float32(0.0001) # The first condition
        mse = get_quant_error(matrix, maxAbs)
        #print("MSE", mse)
        if mse < bestMSE:
            bestMSE = mse
            bestFactor = maxAbs
            increases = 0
        else:
            increases = increases + 1

    # Now that we found the best matrix, replace the Maximum (or minimum number) in the matrix with that number
    # so that it can be nicely picked up by our marian implementation

    maxNum = max(matrix.min(), matrix.max(), key=abs)
    if (abs(maxNum) != bestFactor):
        if maxNum > 0:
            print("MaxAbs before:", maxNum, "now:", bestFactor)
            return np.where(matrix == maxNum, bestFactor, matrix)
        else:
            print("MaxAbs before:", maxNum, "now:", -bestFactor)
            return np.where(matrix == maxNum, -bestFactor, matrix)
    else:
        return matrix

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage:", sys.argv[0], "input_model output_model")
        sys.exit(1)

    model_file = np.load(sys.argv[1])
    model_file_dict = dict(model_file)

    for matrix in model_file_dict.keys():
        if matrix[-2] == 'W' or matrix == "Wemb":
            print(matrix)
            model_file_dict[matrix] = find_best_matrix(model_file_dict[matrix])


    # Save
    np.savez(sys.argv[2], **model_file_dict)
