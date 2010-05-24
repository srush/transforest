#!/usr/bin/env python

import sys

import gflags as flags
FLAGS=flags.FLAGS

if __name__ == "__main__":

    flags.DEFINE_string("lm", None, "SRILM language model file")
    flags.DEFINE_integer("order", 3, "language model order")
    
    

