#!/usr/bin/env python

''' remove spurious nodes in the forest '''

import sys
import time

from forest import *
from node_and_hyperedge import *
from prune import prune

import mymonitor

def remove_eq_edges(edges):
	''' remove equivalent, but sub-optimal edges.
	    edges is a list of (fv, tails) pair
	'''
	
	edges.sort() ## fine
	for i, edge in enumerate(edges):
		if i > 1:
			lastedge = edges[i-1]
			if edge[1] == lastedge[1]:  ## tails equal
				print >> logs, i, edge[0][0], " ".join(map(str, edge[1]))
				print >> logs, i-1, lastedge[0][0], " ".join(map(str, lastedge[1]))
				edges[i] = None			
			
	return [edge for edge in edges if edge is not None]
	
def remove(forest, remove_eq=False):

	start_time = time.time()
	oldsize = forest.size()
	rm = 0

	## alternate TOP-level spuriousness
	## original: TOP* -> TOP  -> S
	## now:      TOP  -> TOP* -> S
	forest.root._spurious = False
	for edge in forest.root.edges:
		edge.subs[0]._spurious = True
	
	for node in forest:

		newedges = []		
		for edge in node.edges:
			nowlist = [(edge.fvector, [])]
			for sub in edge.subs:
				'''muliplication'''
				if sub.is_spurious():
					newlist = []
					for real in sub.edges:
						realsub = real.subs[0]
						for fv, tails in nowlist:
							newtails = tails + [realsub]
							newfv = fv + real.fvector  ## cost

							newlist.append((newfv, newtails))

					nowlist = newlist
				else:
					'''normal nodes'''
					nowlist = [(fv, tails + [sub]) for (fv, tails) in nowlist]
					
			newedges.extend(nowlist)			

		if remove_eq:
			oldlen = len(newedges)
			newedges = remove_eq_edges(newedges)
			rm += oldlen - len(newedges)
			
		node.edges = [Hyperedge(node, tails, fv) for (fv, tails) in newedges]

	forest.nodes = dict([(node.iden, node) for node in forest if not node.is_spurious() or node.is_root()])
	forest.nodeorder = [node for node in forest.nodeorder if not node.is_spurious() or node.is_root()]

	forest.rehash() ## important update for various statistics
	
	newsize = forest.size()
	print >> logs, "%s removed %d nodes and %d edges" % \
		  (forest.tag, oldsize[0] - newsize[0], oldsize[1] - newsize[1]),
	print >> logs, "in %.2lf secs" % (time.time() - start_time)
	print >> logs, "*** removed %d redundant edges!" % rm if rm > 0 else ""
	

if __name__ == "__main__":

	try:
		import psyco
		psyco.full()
	except:
		pass
	
	import optparse
	optparser = optparse.OptionParser(usage="usage: cat <forests> | %prog [options (-h for details)]")
	optparser.add_option("-s", "--suffix", dest="suffix", help="dump suffix (1.suffix)", metavar="SUF")
	optparser.add_option("-S", "--startid", dest="startid", help="%d.suffix start from", \
						 metavar="ID", default=1, type=int)
	optparser.add_option("-q", "--quiet", dest="quiet", action="store_true", help="no dumping", default=False)
	optparser.add_option("-r", "--remove-eq", dest="remove_eq", action="store_true", \
						 help="remove equivalent but redundant edges", default=False)
	optparser.add_option("-R", "--range", dest="range", \
						 help="test pruning (e.g., 5:15:25)", metavar="RANGE", default=None)

	(opts, args) = optparser.parse_args()

	prange = None
	if opts.range is not None:
		prange = eval("[%s]" % opts.range.replace(":", ","))
		prange.sort(reverse=True)

	if opts.quiet and opts.suffix is not None:
		optparser.error("-q and -s can not be present at the same time.")		

	for i, forest in enumerate(Forest.load("-")):
##		print >> logs, "%.4lf\n%s" % f.bestparse()[:2]
		remove(forest)
##		print >> logs, "%.4lf\n%s" % f.bestparse()[:2]
		if not opts.quiet:
			if opts.suffix is not None:
				forest.dump(open("%d.%s" % (i+opts.startid, opts.suffix), "wt"))
			else:
				forest.dump()

		if prange is not None:			
			for p in prange:
				prune(forest, p)
				## TODO: non suffix (single file) version!
				forest.dump("%d.p%d" % (i+opts.startid, p))

 		if i % 10 == 9:
 			mymonitor.gc_collect()
		
