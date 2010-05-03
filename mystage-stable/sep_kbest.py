#!/usr/bin/env python

''' separate a typical k-best parses file into individual files (1.50best, 2.50best...) '''

import sys

if __name__ == "__main__":

    i = 0
    f = None
    k = 50
    for line in sys.stdin:

        if len(line.split("\t")) == 2:  ## "50    1"
            i += 1
            f = open("%d.%dbest" % (i, k), "w")

        print >> f, line,

