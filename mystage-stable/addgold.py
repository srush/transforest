#!/usr/bin/env python

''' simply add the gold tree and a blank line to the end of a forest
'''

import sys

from readkbest import readonebest
from forest import Forest
from remove_sp import remove

if __name__ == "__main__":
    
    import optparse
    optparser = optparse.OptionParser(usage="usage: cat <forests> | %prog -g <GOLDFILE> [-s <suffix>]")
    optparser.add_option("-g", "--gold", dest="goldfile", \
                         help="gold file", metavar="FILE", default=None)
    optparser.add_option("-q", "--quiet", dest="quiet", action="store_true", help="no dumping", default=False)
    optparser.add_option("-r", "--remove", dest="remove_sp", action="store_true", \
                         help="remove spurious", default=False)
    optparser.add_option("-s", "--suffix", dest="suffix", help="dump suffix (1.suffix)", metavar="SUF")

    (opts, args) = optparser.parse_args()

    if opts.goldfile is None:
        opts.error("must specify gold file")
    else:
        goldtrees = readonebest(opts.goldfile)


    for i, forest in enumerate(Forest.load("-")):
        forest.goldtree = goldtrees.next()
        if opts.remove_sp:
            remove(forest)
        if opts.suffix is not None:
            forest.dump(open("%d.%s" % (i+1, opts.suffix), "wt"))
        elif not opts.quiet:
            forest.dump()

        
        
