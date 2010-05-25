#!/usr/bin/env python

import sys

import gflags as flags
FLAGS=flags.FLAGS

from lmstate import LMState

class Decoder(object):

    def __init__(self):
        pass
    
    def add_state(self, new):
        ''' adding a new state to the appropriate beam, and checking finality. '''
        
        self.beams[new.step].append(new)
        if FLAGS.debuglevel >= 2:
            print >> logs, "state %s added to beam %d" % (new, new.step)
        
        if new.is_final():
            if FLAGS.debuglevel >= 2:
                print >> logs, "new final state!"
            if self.best is None or new.score < self.best.score:
                self.best = new
        
    def beam_search(forest, b=1):

        self.best = None
        
        max_step = len(forest) * 2 + 1
        beams = [None] * max_step
        self.beams = beams
        
        beams[0] = [LMState.initstate()] # initial state

        self.nstates = 0  # space complexity
        self.nedges = 0 # time complexity

        for i in range(0, max_step+1):

            beams[i] = sorted(beams[i])[:b]  # beam pruning
            curr_beam = beams[i]
            
            if FLAGS.debuglevel >= 2:
                print >> logs, "beam %d, %d states" % (i, len(curr_beam))
                print >> logs, "\n".join([str(x) for x in curr_beam])
                print >> logs
                
            if curr_beam == []:
                break

            for old in curr_beam:

                for new in old.predict():
                    self.addstate(new)
                    
                for new in old.complete():
                    self.addstate(new)

        return self.best.score, self.best.trans()
    
if __name__ == "__main__":

    flags.DEFINE_string("lm", None, "SRILM language model file")
    flags.DEFINE_integer("order", 3, "language model order")
    flags.DEFINE_integer("beam", 1, "beam size", short_name="b")
    flags.DEFINE_integer("debuglevel", 2, "debug level")

    for i, forest in enumerate(Forest.load("-", transforest=True)):

        score, trans = beam_search(forest, b=FLAGS.beam)

        print "%.4lf\t%s" % (score, trans)

        
