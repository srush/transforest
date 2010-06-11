#!/usr/bin/env python

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
        
        if new not in beam or new < beam[new]: # TODO
            
            beam[new] = new
            if FLAGS.debuglevel >= 2:
                print >> logs, "state %s added to beam %d" % (new, new.step)

            if new.is_final():
                if FLAGS.debuglevel >= 2:
                    print >> logs, "new final state!"
                if self.best is None or new.score < self.best.score:
                    self.best = new

#         elif new in beam:
#             print >> logs, "WARNING recomb", new.score, beam[new].score
        
    def beam_search(self, forest, b=1):

        self.best = None
        
        max_step = len(forest.sent) * 6 + 1
        print "max", max_step
        beams = defaultdict(dict)
        self.beams = beams
        
        self.add_state(LMState.start_state(forest.root)) # initial state

        self.nstates = 0  # space complexity
        self.nedges = 0 # time complexity

        for i in range(0, max_step+1):

            curr_beam = sorted(beams[i].keys())[:b]  # beam pruning
            
            if FLAGS.debuglevel >= 2:
                print >> logs, "beam %d, %d states" % (i, len(curr_beam))
                print >> logs, "\n".join([str(x) for x in curr_beam])
                print >> logs
                
            for old in curr_beam:
                if not old.is_final():
                    for new in old.predict():
                        self.add_state(new)

                    for new in old.complete():
                        self.add_state(new)

        return self.best.score, self.best.trans()
    
if __name__ == "__main__":

    flags.DEFINE_integer("beam", 1, "beam size", short_name="b")
    flags.DEFINE_integer("debuglevel", 0, "debug level")

    from ngram import Ngram
    from model import Model
    from forest import Forest

    argv = FLAGS(sys.argv)

    weights = Model.cmdline_model()
    lm = Ngram.cmdline_ngram()
    
    LMState.init(lm, weights)

    decoder = Decoder()

    tot_bleu = Bleu()
    tot_score = 0.
    tot_time = 0.
    
    for i, forest in enumerate(Forest.load("-", transforest=True), 1):

        t = time.time()
        score, trans = decoder.beam_search(forest, b=FLAGS.beam)
        t = time.time() - t
        tot_time += t

        tot_score += score
        forest.bleu.rescore(trans)
        tot_bleu += forest.bleu

        print >> logs, ("sent %d, b %d\tscore %.4lf\tbleu+1 %.4lf\tlenratio %.2lf" + \
              "\ttime %.3lf\tlen %-3d  nodes %-4d  edges %-5d") % \
              ((i, FLAGS.beam, score) + forest.bleu.score_ratio() + (t, len(forest.sent)) + forest.size())
        print trans

    print >> logs, "avg %d sentences, b %d\tscore %.4lf\tbleu %.4lf\tlenratio %.2lf\ttime %.3lf" % \
          ((i, FLAGS.beam, tot_score/i) + tot_bleu.score_ratio() + (tot_time/i,))
