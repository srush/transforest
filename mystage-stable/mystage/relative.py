#!/usr/bin/env python

''' relatively-useless pruning, after Jon Graehl.
    also called posterior pruning, used by charniak and klein, but only on nodes.

	basically, it is an inside-outside algorithm, with (+, max) semiring.

	\beta (leaf) = 0	
	\beta (n) = max_{e \in BS(n)} \beta(e)
	\beta (e) = max_{n_i \in tails(e)} \beta(n_i) + w(e)

	(bottom-up of \beta is done in node_and_hyperedge.py: Node.bestparse())

	\alpha (root) = 0
	\merit (e) = \alpha (n) + \beta (e),   n = head(e)	 ## alpha-beta
	\alpha (n_i) max= \merit (e) - \beta (n_i),  for n_i in tails(e)

	\merit (n) = \alpha (n) + \beta (n) = max_{e \in BS(n)} \alphabeta (e)
'''

import sys, time

logs = sys.stderr

from forest import Forest

def inside_outside(forest):

	forest.bestparse() ## inside
	
	forest.root.alpha = 0

	for node in forest.reverse():   ## top-down
		assert hasattr(node, "alpha"), node
		node.merit = node.alpha + node.beta
	
		for edge in node.edges:
			edge.merit = node.alpha + edge.beta
			for sub in edge.subs:
				score = edge.merit - sub.beta
				if not hasattr(sub, "alpha") or score < sub.alpha:
					sub.alpha = score
	
def prune(forest, gap, delete=True, do_merits=True):

	start_time = time.time()

	if do_merits:
		inside_outside(forest)

	oldsize = forest.size()
	
	threshold = forest.root.merit + gap	
	newnodes = {}
	neworder = []

	kleinedges = 0
	for node in forest:
		iden = node.iden
		if node.merit <= threshold:  ## node pruning
			newnodes[iden] = node
			neworder.append(node)
			node.edges = [e for e in node.edges if e.merit <= threshold]  ## edge pruning

		else:
			kleinedges += len(node.edges)
			del node

	del forest.nodes
	del forest.nodeorder
	
	forest.nodes = newnodes
	forest.nodeorder = neworder

	forest.rehash() ## important update for various statistics
	
	newsize = forest.size()
	
	print >> logs, "gap %4.1lf, %4d nodes, %5d edges remained. prune ratio = %4.1lf%%, %4.1lf%% (%4.1lf%%)" \
		  % (gap, newsize[0], newsize[1], \
			 (oldsize[0] - newsize[0])*100.0 / oldsize[0], (oldsize[1] - newsize[1])*100.0 / oldsize[1],\
			 kleinedges*100.0/oldsize[1]),
	print >> logs, "done in %.2lf secs" % (time.time() - start_time)

	#print >> logs, "%.4lf\n%s" % forest.bestparse()[:2]


if __name__ == "__main__":

	import optparse
	optparser = optparse.OptionParser(usage="usage: cat <forest> | %prog -p <GAP> [options (-h for details)]")
	optparser.add_option("-p", "--prob", dest="gap", type=float, help="merit threshold", metavar="PROB_GAP")
	optparser.add_option("", "--id", dest="sentid", type=int, help="sentence id", metavar="ID", default=0)
	optparser.add_option("-s", "--suffix", dest="suffix", help="dump suffix (1.suffix)", metavar="SUF")
	optparser.add_option("-q", "--quiet", dest="quiet", action="store_true", help="no dumping", default=False)

	(opts, args) = optparser.parse_args()

	if opts.gap is None:
		optparser.error("must specify GAP.")

	for i, forest in enumerate(Forest.load("-")):

		prune(forest, opts.gap)

		if not opts.quiet:
			if opts.suffix is not None:
				forest.dump(open("%d.%s" % (i+1, opts.suffix), "wt"))
			else:
				forest.dump()
