#!/usr/bin/env python

from __future__ import division

import sys
import time
from collections import defaultdict

logs = sys.stderr

import gflags as flags
FLAGS=flags.FLAGS

from lmstate import LMState
from bleu import Bleu

class Decoder(object):

    def __init__(self):
        pass
    
    def add_state(self, new):
        ''' adding a new state to the appropriate beam, and checking finality. '''

        beam = self.beams[new.step]

        if new.step > self.max_step:
            self.max_step = new.step

        self.num_edges += 1
        
        if new not in beam or new < beam[new]: # safe
            
            beam[new] = new
            if FLAGS.debuglevel >= 2:
                print >> logs, "adding to beam %d: %s" % (new.step, new)

    def beam_search(self, forest, b=1):

        self.num_states = self.num_edges = 0
        self.final_items = []
        self.best = None
        
        beams = defaultdict(defaultdict) # +inf
        self.beams = beams
        
        self.max_step = -1
        self.add_state(LMState.start_state(forest.root)) # initial state

        self.nstates = 0  # space complexity
        self.nedges = 0 # time complexity

        i = 0
        while i <= self.max_step:

            # N.B.: values, not keys! (keys may not be updated)
            curr_beam = sorted(beams[i].values())[:b]  # beam pruning
            self.num_states += len(curr_beam)
            
            if FLAGS.debuglevel >= 1:
                print >> logs, "beam %d, %d states" % (i, len(curr_beam))
                print >> logs, "\n".join([str(x) for x in curr_beam])
                print >> logs
                
            for old in curr_beam:
                if old.is_final():
                    self.final_items.append(old)        

                if not old.is_final():
                    for new in old.predict():
                        self.add_state(new)

                    if FLAGS.complete:
                        for new in old.complete():
                            self.add_state(new)

            i += 1

        self.final_items.sort()
        
        return self.final_items[0], self.final_items[:b]

### main ###

def main():
    
    weights = Model.cmdline_model()
    lm = Ngram.cmdline_ngram()
    
    LMState.init(lm, weights)

    decoder = Decoder()

    tot_bleu = Bleu()
    tot_score = 0.
    tot_time = 0.
    tot_len = tot_fnodes = tot_fedges = 0
    tot_steps = tot_states = tot_edges = 0
    
    for i, forest in enumerate(Forest.load("-", transforest=True), 1):

        t = time.time()
        
        best, final_items = decoder.beam_search(forest, b=FLAGS.beam)
        score, trans, fv = best.score, best.trans(), best.fvector

        t = time.time() - t
        tot_time += t

        tot_score += score
        forest.bleu.rescore(trans)
        tot_bleu += forest.bleu

        fnodes, fedges = forest.size()

        tot_len += len(forest.sent)
        tot_fnodes += fnodes
        tot_fedges += fedges
        tot_steps += decoder.max_step
        tot_states += decoder.num_states
        tot_edges += decoder.num_edges

        print >> logs, ("sent %d, b %d\tk %d\tscore %.4lf\tbleu+1 %s" + \
              "\ttime %.3lf\tsentlen %-3d fnodes %-4d fedges %-5d\tstep %d  states %d  edges %d") % \
              (i, FLAGS.beam, len(final_items), score, 
               forest.bleu.score_ratio_str(), t, len(forest.sent), fnodes, fedges,
               decoder.max_step, decoder.num_states, decoder.num_edges)
        
        if FLAGS.mert: # <score>... <hyp> ...
            print >> logs, '<sent No="%d">' % i
            print >> logs, "<Chinese>%s</Chinese>" % " ".join(forest.cased_sent)

            for item in final_items:
                print >> logs, "<score>%.3lf</score>" % item.score
                print >> logs, "<hyp>%s</hyp>" % item.trans()
                print >> logs, "<cost>%s</cost>" % item.fvector

            print >> logs, "</sent>"
            
        print trans

    print >> logs, ("avg %d sentences, b %d\tscore %.4lf\tbleu %s\ttime %.3f" + \
          "\tsentlen %.1f fnodes %.1f fedges %.1f\tstep %.1f states %.1f edges %.1f") % \
          (i, FLAGS.beam, tot_score/i, tot_bleu.score_ratio_str(), tot_time/i,
           tot_len/i, tot_fnodes/i, tot_fedges/i,
           tot_steps/i, tot_states/i, tot_edges/i)

if __name__ == "__main__":

    from ngram import Ngram
    from model import Model
    from forest import Forest

    flags.DEFINE_integer("beam", 1, "beam size", short_name="b")
    flags.DEFINE_integer("debuglevel", 0, "debug level")
    flags.DEFINE_boolean("mert", True, "output mert-friendly info (<hyp><cost>)")
    flags.DEFINE_boolean("profile", False, "profiling")

    argv = FLAGS(sys.argv)

    if FLAGS.profile:
        import cProfile as profile
        profile.run('main()', '/tmp/a')
        import pstats
        p = pstats.Stats('/tmp/a')
        p.sort_stats('time').print_stats(20)

    else:
        main()

