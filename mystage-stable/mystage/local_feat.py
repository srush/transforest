#!/usr/bin/env python

''' 1. annotate a forest with local features (node-local or edge-local)
    2. split nodes by annotating parent/other info.
'''

import sys
import time

from features import *
from forest import Forest
from utility import *
import heads

debug = False

def print_fnames(fs):
	if len(fs) > 0:
		print "\n".join(["\n".join((name,) * count) for (name, count) in fs])
		
def local_feats(forest, fclasses):

	for node in forest:
		## you will have to annotate parentlabel and heads as soon as you get a tree node by assembling
		node.parentlabel = None		
		
		if not node.is_spurious():  ## neglect spurious nodes
			nodefvector = FVector()
			
			for feat in fclasses:
				if feat.is_nodelocal():
					fs = feat.extract(node, forest.sent)
					if opts.extract:
						print_fnames(fs) 
					else:
						nodefvector += FVector.convert_fullname(fs)		

			node.fvector += nodefvector
##			print >> logs, "%s -------\t%s" % (node, node.fvector)

			for edge in node.edges:
				
				if debug:
					print >> logs, "--------", edge.shorter()
					
				edgefvector = FVector()
				node.subs = edge.subs
				node.rehash()
				
				for feat in fclasses:
					if feat.is_edgelocal():
						fs = feat.extract(node, forest.sent)
						if opts.extract:
							print_fnames(fs)
						else:
							edgefvector += FVector.convert_fullname(fs)
						
				edge.fvector += edgefvector
				if debug and len(edge.fvector) > 1:
					print >> logs, "%s ---------\t%s" % (edge, edge.fvector.pp(usename=True))
	

if __name__ == "__main__":

	try:
		import psyco
		psyco.full()
	except:
		pass
	
	import optparse
	optparser = optparse.OptionParser(usage="usage: cat <forests> | %prog [options (-h)] [<feats>]")
	optparser.add_option("-s", "--suffix", dest="suffix", help="dump suffix (1.suffix)", metavar="SUF")
	optparser.add_option("-q", "--quiet", dest="quiet", action="store_true", help="no dumping", default=False)
	optparser.add_option("-d", "--debug", dest="debug", action="store_true", help="show debug", default=False)
	optparser.add_option("-e", "--extract", dest="extract", action="store_true", \
						 help="extract features names", default=False)

	(opts, args) = optparser.parse_args()
	debug = opts.debug

	fclasses = prep_features(args, read_names=(not opts.extract))
	print >> logs, "features classes", fclasses

	start = time.time()
	for i, forest in enumerate(Forest.load("-")):

		local_feats(forest, fclasses)
		if not opts.quiet and not opts.extract:
			if opts.suffix is not None:
				forest.dump(open("%d.%s" % (i+1, opts.suffix), "wt"))
			else:
				forest.dump()

	total_time = time.time() - start
	print >> logs, "overall: %d sents. local features extracted in %.2lf secs (avg %.2lf per sent)" % \
		  (i+1, total_time, total_time/(i+1))
