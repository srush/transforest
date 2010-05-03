#!/usr/bin/env python

''' read in k-best trees from stdin (in EC format) and compute oracle scores
'''

import sys
import math

logs = sys.stderr

max_k = 100

debug = True

from tree import Tree

from readkbest import readkbest, readonebest
from utility import getfile
from parseval import Parseval
    
if __name__ == "__main__":

    kbestfilename = sys.argv[1]
    goldfilename = sys.argv[2]
    
    kbesttrees = readkbest(kbestfilename)
    print type(kbesttrees), type(kbesttrees.next())

    goldtrees = readonebest(goldfilename)

#     assert len(kbesttrees) == len(goldtrees), "unmatched number of sentences: %d test vs. %d gold" \
#            % (len(kbesttrees), len(goldtrees))


    onebest = Parseval()
    oracle = Parseval()
    
    for (i,  goldtree) in enumerate(goldtrees):

        print >> logs, i,
        print >> logs, goldtree
        _, klist = kbesttrees.next()  # generator

        for (k, (logprob, testtree)) in enumerate(klist):   ## generator again

##            print k, logprob, testtree
##             assert testtree.is_same_sentence(goldtree), "unmatched sentences!\n%s\n%s" \
##                    % (testtree.get_sent(), goldtree.get_sent())

            if k >= max_k:
                continue
    
            result = Parseval(testtree, goldtree)

##            print k, result.brs()
            if k == 0:
                onebest += result
                best = result #.copy()
                best_k = 0
                firstlogp = logprob
            else:
                 if result < best:
##                    print "better f-score! old=", best_fscore
                     best = result #.copy()
                    best_k = k
                lastlogp = logprob

        oracle += best

        if debug: #\t%lf" 
            print " %d\t%d\t%d\t%d\t%lf\t\t%d\t%d\t%lf" \
                  % (i+1, best.goldbr, best.testbr, best.matchbr, best.fscore(), \
                     best_k+1, len(testtree), -lastlogp + firstlogp)

    print onebest
    print oracle
