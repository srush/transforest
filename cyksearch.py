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

import math

class CYKDecoder(object):

    def __init__(self, weights, lm):
        self.weights = weights
        self.lm = lm

    def beam_search(self, forest, b):
        self.translate(forest.root, b)
        return forest.root.hypvec[0]
        
    def translate(self, cur_node, b):
        for hedge in cur_node.edges:
            for sub in hedge.subs:
                if not hasattr(sub, 'hypvec'):
                    self.translate(sub, b)
        # create cube
        if FLAGS.debuglevel > 0:
            print "searching on node: %s" % cur_node
        cands = self.init_cube(cur_node)
        heapq.heapify(cands)
        # gen kbest
        cur_node.hypvec = self.lazykbest(cands, b)
        if FLAGS.debuglevel > 0:
            for (sc, tran, fv) in cur_node.hypvec:
                print tran
            print "_____________________________________"
        
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
        while len(hypvec) < k:
            if cands == []:
                break
            (chyp, cedge, cvecj) = heapq.heappop(cands)
            hypvec.append(chyp)
            self.lazynext(cedge, cvecj, cands)
        #sort and combine hypevec
        hypvec = sorted(hypvec)
        # COMBINATION
        keylist = []
        newhypvec = []
        for (sc, trans, fv) in hypvec:
            if trans not in keylist:
                keylist.append(trans)
                newhypvec.append((sc, trans, fv))
        return newhypvec
    
    def lazynext(self, cedge, cvecj, cands):
        for i in xrange(cedge.arity()):
            ## vecj' = vecj + b^i (just change the i^th dimension
            newvecj = cvecj[:i] + (cvecj[i]+1,) + cvecj[i+1:]
            if newvecj not in cedge.oldvecs:
                newhyp = self.gethyp(cedge, newvecj)
                if newhyp is not None:
                    cedge.oldvecs.append(newvecj)
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
    flags.DEFINE_integer("kbest", 1, "kbest output", short_name="k")

    argv = FLAGS(sys.argv)

    weights = Model.cmdline_model()
    lm = Ngram.cmdline_ngram()

    decoder = CYKDecoder(weights, lm)

    tot_bleu = Bleu()
    tot_score = 0.
    tot_time = 0.
    tot_len = tot_fnodes = tot_fedges = 0

    if FLAGS.debuglevel > 0:
        print "beam size = %d" % FLAGS.beam

    for i, forest in enumerate(Forest.load("-", is_tforest=True, lm=lm), 1):

        t = time.time()
        
        (score, trans, fv) = decoder.beam_search(forest, b=FLAGS.beam)
        print trans
 #       print "best score %s, trans: %s, features: %s" % (score, trans, fv)
        t = time.time() - t
        tot_time += t

        tot_score += score
        forest.bleu.rescore(trans)
        tot_bleu += forest.bleu

        fnodes, fedges = forest.size()

        tot_len += len(forest.sent)
        tot_fnodes += fnodes
        tot_fedges += fedges

#        print >> logs, ("sent %d, b %d\tscore %.4f\tbleu+1 %s" + \
#              "\ttime %.3f\tsentlen %-3d fnodes") % \
#              (i, FLAGS.beam, score, 
#               forest.bleu.score_ratio_str(), t, len(forest.sent), fnodes, fedges)
                                                                           
#    print >> logs, ("avg %d sentences, b %d\tscore %.4lf\tbleu %s\ttime %.3f" + \
#          "\tsentlen %.1f fnodes %.1f fedges") % \
#          (i, FLAGS.beam, tot_score/i, tot_bleu.score_ratio_str(), tot_time/i,
#          tot_len/i, tot_fnodes/i, tot_fedges/i)

