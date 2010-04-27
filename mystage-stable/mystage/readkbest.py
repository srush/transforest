#!/usr/bin/env python

''' read in k-best trees from stdin (in EC format) and rerank them
'''

import sys
import gc
import time

logs = sys.stderr

from tree import Tree
from utility import getfile
from features import *
from parseval import Parseval

class NBestForest(object):

	def __init__(self, k, tag, kparses, goldtree=None):
		self.k = k
		self.tag = tag
		self.kparses = kparses
		self.goldtree = goldtree
		if goldtree:
			self.sent = goldtree.get_sent()

	def oracle(self):
		bestparseval = None 
		for i, (fvector, tree) in enumerate(self.kparses):
			parseval = Parseval (tree, self.goldtree)
			if bestparseval is None or parseval < bestparseval:  ## N.B.: < means better (> in fscore)
				bestparseval = parseval
				assert 0 in fvector, "bad! at k=%d\n%s" % (i, tree)
				bestscore = fvector[0] 
				besttree = tree
				bestfvector = fvector
				
			if opts.k is not None and i+1 >= opts.k:
				break

		return bestscore, bestparseval, besttree, bestfvector

	@staticmethod
	def load(filename, read_gold=True):
		return readkbest(filename, read_gold)

	def dump(self, filename):

		out = open(filename, "wt") if type(filename) is str else filename
		
		print >> out, "%d\t%s" % (len(self.kparses), self.tag)
		for (fv, tr) in self.kparses:
			print >> out, fv
			print >> out, tr			

		print >> out, self.goldtree		

def readkparses(f, k):
	for j in xrange(k):
		fvector = FVector.parse(f.readline().strip())   #float(f.readline().strip())
		parse = f.readline().strip()
		tree = Tree.parse(parse, trunc=True, lower=True)
		
		yield (fvector, tree)				
				
def readkbest(f, read_gold=False):

	f = getfile(f)
	while True: #now < len(lines):
		line = f.readline() #lines[now]
		if line == '':
			break
		if line == '\n':
			continue
		try:
			k, tag = line.strip().split("\t")
			k = int(k)
		except:
			break  ## can finish earlier

		kparses = []
		for stuff in readkparses(f, int(k)):
			kparses.append(stuff)
			
		goldtree = Tree.parse(f.readline().strip(), trunc=True, lower=True) if read_gold \
				   else None
 		yield NBestForest(k, tag, kparses, goldtree)

def readonebest(f):
	'''1-best output, or gold'''
							 
	f = getfile(f)
	while True:
		line = f.readline()
		if line == '':
			break
		if line == '\n':
			continue

		yield Tree.parse(line.strip(), trunc=True, lower=True)


if __name__ == "__main__":
	try:
		import psyco
		psyco.full()
	except:
		pass

	import optparse
	optparser = optparse.OptionParser(usage="usage: cat <nbestlists> | %prog [options (-h for details)]")
	optparser.add_option("-k", "", dest="k", type=int, help="k-best", metavar="K", default=None)
	optparser.add_option("", "--thres", dest="threshold", type=float, \
						 help="threshold/margin", metavar="THRESHOLD", default=None)
	optparser.add_option("-c", "--cross-lines", dest="cross_lines", action="store_true", \
						 help="print features one at a line", default=False)
	optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", \
						 help="print for each sentence", default=False)
	optparser.add_option("-f", "--print-features", dest="print_features", action="store_true", \
						 help="print all features (id=val)", default=False)
	optparser.add_option("-F", "--first", dest="first", type=int, metavar="NUM", \
						 help="only work on the first NUM sentences", default=None)
	optparser.add_option("-O", "--oracle", dest="oracle", action="store_true", \
						 help="compute nbest oracles", default=False)
	optparser.add_option("-g", "--gold", dest="goldfile", \
						 help="gold file", metavar="FILE", default=None)
	optparser.add_option("-s", "--suffix", dest="suffix", \
						 help="dump to $i.suffix", metavar="SUF", default=None)
	
	(opts, args) = optparser.parse_args()

	if opts.goldfile is not None:
		goldtrees = readonebest(opts.goldfile)
	else:
		optparser.error("must specifiy gold trees file (-g FILE)")
		
	if not opts.oracle:

		# extract features, and output to nbestlist format

		fclasses = prep_features(args, read_names=True)

		start_time = time.time()
		extract_time = 0
		all_pp = Parseval()
		for i, forest in enumerate(NBestForest.load("-", read_gold=False)):#(opts.goldfile is None))):

			forest.goldtree = goldtrees.next()

			goldspans = Parseval.parseval_spans(forest.goldtree)

			forest.goldsize = len(goldspans)
			
			print "%s\t%d\t%d" % (forest.tag, forest.goldsize, forest.k)

			best_pp = None               ## CAREFUL! could be 0
			for k, (fvector, tree) in enumerate(forest.kparses):
				tree.annotate_all()

				extract_time -= time.time()
				newfv = extract(tree, tree.get_sent(), fclasses)
				fvector += newfv				
				extract_time += time.time()

				pp = Parseval(tree, forest.goldtree)
				print "#%d\t%d %d\t%s" % (k, pp.matchbr, pp.testbr, fvector)

				if best_pp is None or pp < best_pp: ## SMALL IS BETTER IN PARSEVAL
					best_pp = pp
					best_k = k

			print >> logs, i+1, "\t", best_pp
			all_pp += best_pp

			if opts.first is not None and i+1 >= opts.first:
				break

		total_time = time.time() - start_time
		print >> logs, "%d sentences extracted in %.2lf secs (extract time %.2lf secs)" % \
			  (i+1, total_time, extract_time)
		print >> logs, all_pp		

# 			if opts.suffix is not None:
				
# 				outfile = "%d.%s" % (i+1, opts.suffix)
# 				forest.dump(outfile)
# 				print >> logs, "forest #%d dumped to %s" % (i+1, outfile)
# ##			else:
# ##				forest.dump(sys.stdout)

	else:
		## compute nbest oracles
		parseval = Parseval()
		for i, forest in enumerate(NBestForest.load("-", read_gold=(opts.goldfile is None))):

			if opts.goldfile is not None:
				forest.goldtree = goldtrees.next()
				
			score, forest.oracleparseval, forest.oracletree, forest.fvector = forest.oracle()
##			print forest.oracletree

			if opts.verbose:
				print "%d\t%s" % (i+1, forest.oracleparseval)
				
			parseval += forest.oracleparseval 

			if opts.first is not None and i+1 >= opts.first:
				break

		print parseval
			

			
