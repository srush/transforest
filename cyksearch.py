#!/usr/bin/env python

import sys
import time
from collections import defaultdict

logs = sys.stderr

import gflags as flags
FLAGS=flags.FLAGS

from bleu import Bleu
from svector import Vector

import heapq

class CYKDecoder(object):

    def __init__(self, weights, lm):
        self.weights = weights
        self.lm = lm

    def beam_search(self, forest, b):
        self.translate(forest.root, b)
        self.lm_edges = 0
        self.lm_nodes = 0
        return forest.root.hypvec[0]

    def search_size(self):
        return self.lm_nodes, self.lm_edges

    def reset_size(self):
        self.lm_nodes = 0
        self.lm_edges = 0
    
    def translate(self, cur_node, b):
        for hedge in cur_node.edges:
            for sub in hedge.subs:
                if not hasattr(sub, 'hypvec'):
                    self.translate(sub, b)
        # create cube
        cands = self.init_cube(cur_node)
        heapq.heapify(cands)
        # gen kbest
        cur_node.hypvec = self.lazykbest(cands, b)

        #lm nodes
        self.lm_nodes += len(cur_node.hypvec)
        #lm edges
        for cedge in cur_node.edges:
            self.lm_edges += len(cedge.oldvecs)
 
        
        
    def init_cube(self, cur_node):
        cands = []
        for cedge in cur_node.edges:
            cedge.oldvecs = set()
            newvecj = (0,) * cedge.arity()
            cedge.oldvecs.add(newvecj)
            newhyp = self.gethyp(cedge, newvecj)
            cands.append((newhyp, cedge, newvecj))
        return cands

        
    def lazykbest(self, cands, k):
        hypvec = []
        if FLAGS.cube:
            signs = set()
            cur_kbest = 0
            while cur_kbest < k:
                if cands == []:
                    break
                (chyp, cedge, cvecj) = heapq.heappop(cands)

                cursig = CYKDecoder.gen_sign(chyp[1])
                if cursig not in signs:
                    signs.add(cursig)
                    cur_kbest += 1
                    
                hypvec.append(chyp)
                self.lazynext(cedge, cvecj, cands)
        else:
            while True:
                if cands == []:
                    break
                (chyp, cedge, cvecj) = heapq.heappop(cands)
                hypvec.append(chyp)
                self.lazynext(cedge, cvecj, cands)

        #sort and combine hypevec
        hypvec = sorted(hypvec)[0:k]
        #COMBINATION
        keylist = set()
        newhypvec = []
        for (sc, trans, fv) in hypvec:
            if trans not in keylist:
                keylist.add(trans)
                newhypvec.append((sc, trans, fv))
        
        return newhypvec
    
    def lazynext(self, cedge, cvecj, cands):
        for i in xrange(cedge.arity()):
            ## vecj' = vecj + b^i (just change the i^th dimension
            newvecj = cvecj[:i] + (cvecj[i]+1,) + cvecj[i+1:]
            if newvecj not in cedge.oldvecs:
                newhyp = self.gethyp(cedge, newvecj)
                if newhyp is not None:
                    cedge.oldvecs.add(newvecj)
                    heapq.heappush(cands, (newhyp, cedge, newvecj))
                        
    def gethyp(self, cedge, vecj):
        score = cedge.fvector.dot(self.weights) 
        fvector = Vector(cedge.fvector)
        subtrans = []
        lmstr = cedge.lhsstr

        for i, sub in enumerate(cedge.subs):
            if vecj[i] >= len(sub.hypvec):
                return None
            (sc, trans, fv) = sub.hypvec[vecj[i]]
            subtrans.append(trans)
            score += sc
            fvector += fv
        
        (lmsc, alltrans) = CYKDecoder.deltLMScore(lmstr, subtrans)
        score += (lmsc * self.weights['lm'])  
        fvector['lm'] += lmsc
        return (score, alltrans, fvector)
    
    @staticmethod
    def get_history(history):
        return ' '.join(history[-lm.order+1:] if len(history) >= lm.order else history)

    @staticmethod
    def gen_sign(self, trans):
        if len(trans) >= lm.order:
            return ' '.join(trans[:lm.order-1]) + ' '.join(trans[-lm.order+1:])
        else:
            return ' '.join(trans)
                            
    @staticmethod
    def deltLMScore(lhsstr, sublmstr):
        history = []
        lmscore = 0.0
        nextsub = 0
        for lstr in lhsstr:
            if type(lstr) is str:  # it's a word
                lmscore += lm.word_prob_bystr(lstr, CYKDecoder.get_history(history))
                history.append(lstr)
                
            else:  # it's variable
                curtrans = sublmstr[nextsub].split()
                nextsub += 1
                if len(history) == 0: # the beginning words
                    history.extend(curtrans)
                else:
                    # TODO: minus prob
                    for i, word in enumerate(curtrans, 1):
                        if i < lm.order:
                            lmscore += lm.word_prob_bystr(word,\
                                                   CYKDecoder.get_history(history))
                            # minus the P(w1) and P(w2|w1) ..
                            myhis = ' '.join(history[-i+1:]) if i>1 else ''
                            lmscore -= lm.word_prob_bystr(word, myhis)
                            
                        history.append(word)
                        
        return (lmscore, " ".join(history))

if __name__ == "__main__":

    from ngram import Ngram
    from model import Model
    from forest import Forest

    flags.DEFINE_integer("beam", 1, "beam size", short_name="b")
    flags.DEFINE_integer("debuglevel", 0, "debug level")
    flags.DEFINE_boolean("mert", True, "output mert-friendly info (<hyp><cost>)")
    flags.DEFINE_boolean("cube", True, "using cube pruning to speedup")
    flags.DEFINE_integer("kbest", 1, "kbest output", short_name="k")

    argv = FLAGS(sys.argv)

    weights = Model.cmdline_model()
    lm = Ngram.cmdline_ngram()

    decoder = CYKDecoder(weights, lm)

    tot_bleu = Bleu()
    tot_score = 0.
    tot_time = 0.
    tot_len = tot_fnodes = tot_fedges = 0

    tot_lmedges = 0
    tot_lmnodes = 0
    if FLAGS.debuglevel > 0:
        print >>logs, "beam size = %d" % FLAGS.beam

    for i, forest in enumerate(Forest.load("-", is_tforest=True, lm=lm), 1):

        t = time.time()

        # set the lm_nodes and lm_edges to zero
        decoder.reset_size()
        #decoding
        (score, trans, fv) = decoder.beam_search(forest, b=FLAGS.beam)

        t = time.time() - t
        tot_time += t

        print trans
        print >>logs, "featurs: %s" % fv
        
        tot_score += score
        forest.bleu.rescore(trans)
        tot_bleu += forest.bleu

        # lm nodes and edges
        lm_nodes, lm_edges = decoder.search_size()
        tot_lmnodes += lm_nodes
        tot_lmedges += lm_edges
        
        # tforest size
        fnodes, fedges = forest.size()
        tot_fnodes += fnodes
        tot_fedges += fedges

        tot_len += len(forest.sent)

        print >> logs, "sent %d, b %d\tscore:%.3lf time %.3lf\tsentlen %d\tfnodes %d\ttfedges %d\tlmnodes %d\tlmedges %d" % \
              (i, FLAGS.beam, score, t, len(forest.sent), fnodes, fedges, lm_nodes, lm_edges)
                                                                           
    print >> logs, ("avg %d sentences, b %d\tscore %.4lf\ttime %.3lf" + \
                    "\tsentlen %.3lf fnodes %.1lf fedges %.1lf" + \
                    "\tlmnodes %.3lf lmedges %.3lf") % \
                    (i, FLAGS.beam, float(tot_score)/i, tot_time/i, \
                     tot_len/i, tot_fnodes/i, tot_fedges/i,\
                     tot_lmnodes/i, tot_lmedges/i)

