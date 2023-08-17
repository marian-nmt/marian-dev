#!/usr/bin/env python3
"""
This script converts *.chpt files to *.pt files, potentially useful for extracting weights only from larger checkpoints.
"""

import torch
import argparse

# Create a parser for command line arguments
parser = argparse.ArgumentParser()

# Add arguments for the source and target files
parser.add_argument("--source", type=str, required=True, help="Path to the source *.chpt file")
parser.add_argument("--target", type=str, required=True, help="Path to the target *.pt file")

# Parse the command line arguments
args = parser.parse_args()

# Load the model from the source file
model = torch.load(args.source)

# Save the model to the target file
torch.save(model, args.target)