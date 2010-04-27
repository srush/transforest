#!/usr/bin/env python

''' read in k-best trees from stdin (in EC format) and rerank them

for n-best oracle
> cat 23.50best | readkbest.py -g ../../wsj-data/23.cleangold -O [-v] [-k<K>] [-F<first>]

2416 sents, prec 42708/44022 0.9702     recall 42708/44276 0.9646       f-score 0.9674 complete 0.711

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

		ret =  (fvector, tree)				
		yield ret
				
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

		ret = NBestForest(k, tag, kparses, goldtree)
 		yield ret

def readonebest(f):
	'''1-best output, or gold'''
							 
	f = getfile(f)
	while True:
		line = f.readline()
		if line == '':
			break
		if line == '\n':
			continue

		ret = Tree.parse(line.strip(), trunc=True, lower=True)
		yield ret


if __name__ == "__main__":
	try:
		import psyco
		psyco.full()
	except:
		pass

	import optparse
	optparser = optparse.OptionParser(usage="output feature counts on n-best parses in a compact form, including parsevals.\nusage: cat <nbestlists> | %prog [options (-h for details)] [features]")
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
	optparser.add_option("", "--nologprob", dest="nologprob", action="store_true", \
						 help="do not include log prob (default: include)", default=False)
	optparser.add_option("-A", "--absolute", dest="reduce", action="store_false", \
						 help="use absolute instead of relative counts (default: reduced)", default=True)
	optparser.add_option("-g", "--gold", dest="goldfile", \
						 help="gold file", metavar="FILE", default=None)
	optparser.add_option("-m", "--merge", dest="mergewith", \
						 help="merge with (append to) another nbest features file", metavar="FILE", default=None)
	optparser.add_option("-s", "--suffix", dest="suffix", \
						 help="dump to $i.suffix", metavar="SUF", default=None)
	
	(opts, args) = optparser.parse_args()

	from nbestdecoder import reduce_counts
	
	if opts.goldfile is not None:
		goldtrees = readonebest(opts.goldfile)
	else:
		if not opts.print_features:
			optparser.error("must specifiy gold trees file (-g FILE)")

	if opts.mergewith is not None:
		from nbestdecoder import NBestList
		old_nbestlists = NBestList.load(opts.mergewith)
		opts.nologprob = True


	if opts.print_features:
		opts.reduce = False
		
	if not opts.oracle:

		# extract features, and output to nbestlist format

		fclasses = prep_features(args, read_names=True)

		start_time = time.time()
		extract_time = 0
		all_pp = Parseval()
		for i, forest in enumerate(NBestForest.load("-", read_gold=False)):#(opts.goldfile is None))):

			if not opts.print_features:
				forest.goldtree = goldtrees.next()
				goldspans = Parseval.parseval_spans(forest.goldtree)
				forest.goldsize = len(goldspans)
				print "%s\t%d\t%d" % (forest.tag, forest.goldsize, forest.k)
				best_pp = None               ## CAREFUL! could be 0

			if opts.mergewith is not None:
				oldlist = old_nbestlists.next()
				assert forest.k == oldlist.k, "kbest length mismatch %d : %d" % (oldlist.k, forest.k)
				assert oldlist.goldsize == forest.goldsize, \
					   "gold size mismatch in merge: %d : %d" % (oldlist.goldsize, forest.goldsize)
 
			for k, (fvector, tree) in enumerate(forest.kparses):

				if opts.nologprob:
					del fvector[0]
					
				if opts.mergewith:
					(old_fv, old_pp) = oldlist.kparses[k]
					fvector += old_fv
##					assert old_fv[0] == fvector[0], "logprob mismatch in merge"
##					old_fv[0] = 0 ## N.B.: don't count logprob twice
				
				tree.annotate_all()

				extract_time -= time.time()
				newfv = extract(tree, tree.get_sent(), fclasses)
				fvector += newfv				
				extract_time += time.time()


				if opts.print_features:
					del fvector[0] ## no logprob
					print "#%d" % k, fvector
				else:
					if opts.mergewith:
						pp = old_pp # not checking, save time
					else:
						pp = Parseval(tree, forest.goldtree)

					if not opts.reduce:
						print "#%d\t%d %d\t%s" % (k, pp.matchbr, pp.testbr, fvector)
					else:
						## hacky
						tree.matchbr = pp.matchbr
						tree.testbr = pp.testbr

					if best_pp is None or pp < best_pp: ## SMALL IS BETTER IN PARSEVAL
						best_pp = pp
						best_k = k

			if opts.reduce:
				reduce_counts(forest.kparses)				
				for k, (fvector, tree) in enumerate(forest.kparses):
					print "#%d\t%d %d\t%s" % (k, tree.matchbr, tree.testbr, fvector)

			if not opts.print_features:
				print >> logs, i+1, "\t", best_pp
				all_pp += best_pp

			if opts.first is not None and i+1 >= opts.first:
				break

		total_time = time.time() - start_time
		print >> logs, "%d sentences extracted in %.2lf secs (extract time %.2lf secs)" % \
			  (i+1, total_time, extract_time)

		if not opts.print_features:
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
			

			
