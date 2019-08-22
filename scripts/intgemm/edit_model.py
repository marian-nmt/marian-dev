#!/usr/bin/env python3

import numpy as np
from sys import argv
from typing import Dict, Tuple

def getMaxDict(filename: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    infile = open(filename, 'r')

    vals = {}
    vals_min = {}
    for line in infile:
        splitted = line.strip().split("\t")
        name = ""
        val = 0
        if len(splitted) == 1:
            val = float(splitted[0])
        else:
            name = splitted[0].split("::")[1]
            val = float(splitted[1])

        if name not in vals:
            vals[name] = val
            vals_min[name] = val
        if vals[name] < val:
            vals[name] = val
        if vals_min[name] > val:
            vals_min[name] = val
    return (vals, vals_min)

def addExtraDataToModel(filename: str, outfilename: str, alphadict: Dict[str, float]):
    numpy_mod = dict(np.load(filename))
    for name in alphadict:
        numpy_mod[name + "_alpha"] = np.float32(alphadict[name])

    np.savez(outfilename, **numpy_mod)

if __name__ == '__main__':
    if len(argv) < 4:
        print("Usage: ", argv[0], "matrixStats input.npz output.npz")
        exit()

    max_dict, min_dict = getMaxDict(argv[1])
    addExtraDataToModel(argv[2], argv[3], max_dict)
