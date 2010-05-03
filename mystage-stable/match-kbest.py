#!/usr/bin/env python

''' match a trees file (say, gold, or decoded) against an nbest file and output rank (if found).

cat <nbest_lower_case> | match-kbest.py [-k <MAXK>] <gold_file>

<gold_file> can be gold trees, or trees produced by forest decoding.

'''

import sys

from utility import desymbol, symbol

def normalize(parseline):
    return " ".join(map(lambda x: desymbol(x.lower()), parseline.split()))

if __name__ == "__main__":

    maxk = 0
    out = 0
    hit = {}
    for i in range(1, 101):
        hit[i] = 0

    goldfile = open(sys.argv[1])
    nbestfile = sys.stdin    

    for i, goldline in enumerate(goldfile):

        goldline = normalize(goldline)
        #print goldline

        # k for a sentence: 50\twsj_23.0000
        num = int(nbestfile.readline().strip().split("\t")[0])
        maxk = max(num, maxk)

        matched = False
        for k in range(1, num+1):            
            p = nbestfile.readline()
            if p[0]=="-":
                p = nbestfile.readline() ## skipping the prob line
                
            parseline = normalize(p)
            #if k == 1: print "---", parseline

            if goldline == parseline:
                hit [k] += 1
                matched = True
                #print "%d\t%d" % (i, k)                

        if not matched:
            out += 1
            #print ">", k
        
    total = i
    for k in range(1, maxk+1):
        print "%d\t%.2lf%%" % (k, 100.0 * hit[k] / total)

    print "> %d\t%.2lf%%" % (k, 100.0 * out / total)
