#!/usr/bin/env python

''' 1. annotate a forest with local features (node-local or edge-local)
    2. split nodes by annotating parent/other info.
'''

import sys
from features import *
from forest import Forest
from utility import *
import heads

def local_feats(forest, fclasses):

	for node in forest:
		## you will have to annotate parentlabel and heads as soon as you get a tree node by assembling
		node.parentlabel = None		
		
		if not node.is_spurious():  ## neglect spurious nodes
			nodefvector = FVector()
			
			for feat in fclasses:
				if feat.is_nodelocal():
					nodefvector += FVector.convert_fullname(feat.extract(node, forest.sent))		

			node.fvector += nodefvector

			print "%s -------\t%s" % (node, node.fvector)


			for edge in node.edges:
				
				edgefvector = FVector()
				node.subs = edge.subs
				
				for feat in fclasses:
					if feat.is_edgelocal():
						edgefvector += FVector.convert_fullname(feat.extract(node, forest.sent))
						
				edge.fvector += edgefvector
				print "%s ---------\t%s" % (edge, edge.fvector)
	

if __name__ == "__main__":

	try:
		import psyco
		psyco.full()
	except:
		pass
	
	import optparse
	optparser = optparse.OptionParser(usage="usage: cat <forest> | %prog [options (-h for details)]")
	optparser.add_option("", "--id", dest="sentid", type=int, help="sentence id", metavar="ID", default=0)

	(opts, args) = optparser.parse_args()

	fclasses = prep_features(["word-1", "rule-1", "wordedges"])

	for forest in Forest.load("-"):
		local_feats(forest, fclasses)
	
##		break

##	forest.dump()

