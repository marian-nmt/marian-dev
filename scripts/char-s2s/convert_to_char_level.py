#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert text into char level, e.g. put spaces between every two characters and
replace original spases with a special token (<space>).
"""

import sys
import argparse
import logging

def convert_to_char_level(line, space_sym):
    out = []
    for char in line:
        if char == ' ':
            out.append(space_sym)
        else:
            out.append(char)
    return out


def revert_from_char_level(line, space_sym):
    tokens = []
    for token in line.split(' '):
        if token == space_sym:
            tokens.append(' ')
        else:
            tokens.append(token)
    return ''.join(tokens)

def parse_args():
    parser = argparse.ArgumentParser(description='Convert to char level.')
    parser.add_argument("--space", "-s", default="<space>",
                        help="token replacement for space symbol.")
    parser.add_argument("--revert", "-r", default=False, action='store_true',
                        help="Remove spaces between characters and replace \
                              space replacement character with normal space.")
    return parser.parse_args()

def main():
    """ main """
    args = parse_args()
    space_sym = args.space
    revert = args.revert
    logging.info("Processing to char level input.")
    logging.info("Using {} for space replacement.".format(space_sym))

    for line in sys.stdin:
        line = line.strip()
        if revert:
            print(revert_from_char_level(line, space_sym))
        else:
            print(' '.join(convert_to_char_level(line, space_sym)))

if __name__ == "__main__":
    main()
