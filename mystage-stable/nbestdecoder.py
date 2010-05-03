#!/usr/bin/env python

import sys
import time
import heapq
import math

logs = sys.stderr

from decoder import Decoder
from fvector import FVector
from parseval import Parseval
from utility import *

def reduce_counts(kparses):
    
    overall = FVector()

    for fvector, _ in kparses:
        for f, v in fvector.items():
            overall [f] = overall.get(f, 0) + v

    l = len(kparses)
    for f, v in sorted(overall.items()):
        overall[f] = int(round(float(v)/l - 0.0001))
    ##    print >>logs, f, overall[f], "\t",

    for fvector, _ in kparses:
        for f, v in overall.items():
            if v == 0:
                continue
            x = fvector.get(f, 0) - v
            if math.fabs(x) > 1e-4:
                fvector [f] = x
            else:
                del fvector [f]

class NBestList(object):
    ''' Lazy forest-wise, non-lazy parse-wise.'''

    def __init__(self, k, tag, kparses, goldsize):
        self.k = k
        self.tag = tag
        self.kparses = kparses
        self.goldsize = goldsize

    @staticmethod
    def load(filename):
        '''small.13.1      7       50
           #0      5 7           0=-42.9527 ...
           ...
        N.B.  a dummy TAB between sizes and fvector, sorry.
        '''
        
        total_time = 0
        num_sents = 0
        f = getfile(filename)
        while True: #now < len(lines):

            start_time = time.time()
            
            line = f.readline() #lines[now]
            if line == '':
                break

            num_sents += 1
##            print >> logs, line,
            tag, goldsize, k = line.split("\t")
            goldsize = int(goldsize)
            k = int(k)

            kparses = []
            best_pp = None   ## CAREFUL! could be 0
            for i in xrange(k):
                sentid, sizes, _, fv = f.readline().split("\t")
                matchbr, testbr = map(int, sizes.split())
                fvector = FVector.parse(fv)
                pp = Parseval.get_parseval(matchbr, testbr, goldsize)

                curr = [fvector, pp]
                kparses.append(curr)

                if best_pp is None or pp < best_pp:  ## < is better in oracle
                    best_pp = pp
                    oracle = curr
                    oracle_testbr = testbr

            forest = NBestList(k, tag, kparses, goldsize)
            forest.oracle_tree = oracle
            forest.oracle_fvector, forest.oracle_pp = oracle
            
            if Decoder.MAX_NUM_BRACKETS < 0:
                forest.oracle_size_ratio = 1
            else:
                forest.oracle_size_ratio = oracle_testbr / Decoder.MAX_NUM_BRACKETS

            total_time += time.time() - start_time

            yield forest

        NBestList.load_time = total_time
        print >> logs, "%d nbest lists loaded in %.2lf secs (avg %.2lf per sent)" \
              % (num_sents, total_time, total_time/num_sents)


    def dump(self, out=sys.stdout):

        print >> out, "%s\t%d\t%d" % (self.tag, self.goldsize, self.k)
        for k, (fvector, pp) in enumerate(self.kparses):
            print >> out, "#%d\t%d %d\t%s" % (k, pp.matchbr, pp.testbr, fvector)

    def reduce_counts(self):
        ''' use relative counts instead of absolute, resulting in a much sparser fvector.
            due to johnson.
        '''
        reduce_counts(self.kparses)
        
class NBestDecoder(Decoder):
    
    __str__ = lambda x: "NBestDecoder"

    def __init__(self, N=50):
        Decoder.__init__(self)
        self.N = N

    def load_time(self):
        return NBestList.load_time

    def _decode(self, forest, weights):
        ''' as specified in Decoder.py, should return (score, tree, fvector, parseval) '''
        
        bestscore = None
        
        for i, tree in enumerate(forest.kparses):
            fvector, pp = tree
            score = fvector * weights
            
##            if i == 0:
##                base_score = score
##            else:
##                print >> logs, "%d: %.2lf\t" % (i, score - base_score),  

            if bestscore is None or score < bestscore:
                bestscore = score
                besttree = tree
                bestfvector = fvector
                bestpp = pp
                bestindex = i

            if i+1 >= self.N:
                break

        print >> logs, "best=", bestindex,
        self.bestindex = bestindex ## hacky
        return bestscore, besttree, bestfvector, bestpp

    def _oracle(self, forest):
        ''' did everything in load'''
        pass

    def load(self, filename):        
        return NBestList.load(filename)


if __name__ == "__main__":
    
    import optparse
    optparser = optparse.OptionParser(usage="usage: cat <nbestlists> | %prog [options (-h)]")
    optparser.add_option("-N", "", dest="N", type=int, help="first N-best only", metavar="N", default=50)
    optparser.add_option("-k", "", dest="N", type=int, help="first N-best only", metavar="N", default=50)
    optparser.add_option("-W", "", dest="weightsfile", help="read weights from", metavar="FILE", default=None)
    optparser.add_option("-w", "", dest="weights", help="read weights from str", metavar="W", default=None)
    optparser.add_option("-O", "--oracle", dest="oracle", action="store_true", \
                         help="compute nbest oracles (instead of decoding)", default=False)
    optparser.add_option("-R", "--reduce", dest="reduce", action="store_true", \
                         help="reduce absolute feature counts to relative", default=False)
    optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", \
                         help="print result for each sentence", default=False)
    optparser.add_option("-t", "--trees", dest="nbesttreesfile", help="read nbest trees", \
                         metavar="FILE", default=None)

    (opts, args) = optparser.parse_args()

    if opts.weights:
        weights = FVector.parse(opts.weights)
    elif opts.weightsfile:
        weights = FVector.readweights(opts.weightsfile)
    else:
        weights = FVector({0:1})

    if opts.nbesttreesfile is not None:
        from readkbest import NBestForest
        nbesttrees = NBestForest.load(opts.nbesttreesfile, read_gold=False)        

    decoder = NBestDecoder(opts.N)
    
    all_pp = Parseval()
    decode_time, parseval_time = 0, 0
    
    
    for i, forest in enumerate(decoder.load("-")):

        if opts.oracle:
            all_pp += forest.oracle_pp

        elif opts.reduce:
            forest.reduce_counts()
            forest.dump()
            print >> logs, i+1
            
        else:
            score, tree, fvector, pp = decoder.decode(forest, weights)

            parseval_time -= time.time()            
            all_pp += pp
            parseval_time += time.time()

            if opts.verbose:
                print >> logs, pp

            if opts.nbesttreesfile is not None:
                trees = nbesttrees.next()
                fv, tr = trees.kparses[decoder.bestindex] ## hacky, fv is logprob now
                print tr

    if not opts.reduce:
        print all_pp
