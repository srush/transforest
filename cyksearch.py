#!/usr/bin/env python
from __future__ import division

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
        self.lm_edges = 0
        self.lm_nodes = 0
        # translation
        self.translate(forest.root, b)
        return forest.root.hypvec[0]

    def output_kbest(self, forest, i, mert):
        '''output the kbest translations in root vector'''
        if mert:
            print >> logs, '<sent No="%d">' % i
            print >> logs, "<Chinese>%s</Chinese>" % " ".join(forest.cased_sent)

        for k, (sc, trans, fv) in enumerate(forest.root.hypvec, 1):
            if mert:
                print >> logs, "<score>%.3lf</score>" % sc
                print >> logs, "<hyp>%s</hyp>" % trans
                print >> logs, "<cost>%s</cost>" % fv
                #HM: lm test
                # print >> logs, "lm score confirmation: %.3lf" % lm.hm_word_prob(trans) 

            hyp_bleu = forest.bleu.rescore((trans))
            print >> logs, "k=%d\tscore=%.4lf\tbleu+1=%.4lf\tlenratio=%.2lf\t%s" % \
                  (k, sc, hyp_bleu, forest.bleu.ratio(), fv)

        if mert:
            print >> logs, "</sent>"
            
    def search_size(self):
        return self.lm_nodes, self.lm_edges
   
    def translate(self, cur_node, b):
        for hedge in cur_node.edges:
            for sub in hedge.subs:
                if not hasattr(sub, 'hypvec'):
                    self.translate(sub, b)
        # create cubew
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

        
    def lazykbest(self, cands, b):
        hypvec = []
        signs = set()
        cur_kbest = 0
        while cur_kbest < b:
            if cands == [] or len(hypvec) >= (FLAGS.ratio*b):
                break
            (chyp, cedge, cvecj) = heapq.heappop(cands)
            if chyp[3] not in signs:
                signs.add(chyp[3])
                cur_kbest += 1
            hypvec.append(chyp)
            self.lazynext(cedge, cvecj, cands)
 
        #sort and combine hypevec
        hypvec = sorted(hypvec)
        #COMBINATION
        keylist = set()
        newhypvec = []
        for (sc, trans, fv, sig) in hypvec:
            if sig not in keylist:
                keylist.add(sig)
                newhypvec.append((sc, trans, fv))
            if len(newhypvec) >= b:
                break
        
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
        
        (lmsc, alltrans, sig) = CYKDecoder.deltLMScore(lmstr, subtrans)
        score += (lmsc * self.weights['lm'])  
        fvector['lm'] += lmsc
        return (score, alltrans, fvector, sig)
    
    @staticmethod
    def get_history(history):
        return ' '.join(history[-lm.order+1:] if len(history) >= lm.order else history)

    @staticmethod
    def gen_sign(trans):
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
                    # minus prob
                    for i, word in enumerate(curtrans, 1):
                        if i < lm.order:
                            lmscore += lm.word_prob_bystr(word,\
                                                   CYKDecoder.get_history(history))
                            # minus the P(w1) and P(w2|w1) ..
                            myhis = ' '.join(history[-i+1:]) if i>1 else ''
                            lmscore -= lm.word_prob_bystr(word, myhis)
                            
                        history.append(word)
        
        return (lmscore, " ".join(history), CYKDecoder.gen_sign(history))

if __name__ == "__main__":

    from ngram import Ngram
    from model import Model
    from forest import Forest

    flags.DEFINE_integer("beam", 100, "beam size", short_name="b")
    flags.DEFINE_integer("debuglevel", 0, "debug level")
    flags.DEFINE_boolean("mert", False, "output mert-friendly info (<hyp><cost>)")
    flags.DEFINE_boolean("cube", True, "using cube pruning to speedup")
    flags.DEFINE_integer("kbest", 1, "kbest output", short_name="k")
    flags.DEFINE_integer("ratio", 3, "the maximum items (pop from PQ): ratio*b", short_name="r")
  

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
        #decoding
        (score, trans, fv) = decoder.beam_search(forest, b=FLAGS.beam)

        t = time.time() - t
        tot_time += t

        print trans
        if FLAGS.kbest > 1:
            decoder.output_kbest(forest, i, FLAGS.mert)
        else:
            hyp_bleu = forest.bleu.rescore((trans))
            print >> logs, "k=1 tscore=%.4lf\tbleu+1=%.4lf\tlenratio=%.2lf\t%s" % \
                  (score, hyp_bleu, forest.bleu.ratio(), fv)

        tot_score += score
        #forest.bleu.rescore((trans))
        #tot_bleu += forest.bleu

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

