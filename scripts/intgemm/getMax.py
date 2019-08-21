#!/usr/bin/env python3

from sys import argv
infile = open(argv[1], 'r')

vals = {}
for line in infile:
    splitted = line.strip().split("\t")
    name = ""
    val = 0
    if len(splitted) == 1:
        val = splitted[0]
    else:
        name = splitted[0]
        val = splitted[1]

    if name not in vals:
        vals[name] = val
    if vals[name] < val:
        vals[name] = val
        
for item in vals:
    print(item, "\t", vals[item])
        
