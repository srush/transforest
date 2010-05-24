#!/usr/bin/env python

import sys

logs = sys.stderr

import gflags as flags
FLAGS=flags.FLAGS

flags.DEFINE_string("weights", None, "weights str or filename", short_name="w")

from svector import Vector

class Model(object):

    __slots__ = "weights", "lm_weight"
    
    def __init__(self, w):
        '''input is either a filename or weightstr.'''

        if w.strip() == "":
            self.weights = Vector()
        else:
            if not (w.find(":") >= 0 or w.find("=") >= 0):
                w = open(w).readline().strip() # single line
            self.weights = Vector(w)

        print >> logs, 'using weights: "%s...%s" (%d fields)' \
                    % (w[:10], w[-10:], len(self.weights))

        self.lm_weight = self.weights["lm"]

    @staticmethod
    def cmdline_model():
        if FLAGS.weights is None:
            print >> logs, "Error: must specify weights by -w" + str(FLAGS)
            sys.exit(1)
            
        return Model(FLAGS.weights)
    
if __name__ == "__main__":

    argv = FLAGS(sys.argv)

    Model.cmdline_model()
