#!/usr/bin/env python

''' using forest as a virtual infty-best list (suggested by Mark Johnson).
also used in Forest-based Translation paper.

cat <forest_file> | virtual-kbest.py [-k <MAXK>] <gold_file>

<gold_file> can be gold trees, or trees produced by forest decoding.

'''


import sys, os, re

if __name__ == "__main__":

    goldfile = open(sys.argv[1])
    
    for i, f in enumerate(Forest.load("-")):

        line = goldfile.readline()
        goldtree = Tree.parse(line.strip(), trunc=True, lower=True)

        
        
