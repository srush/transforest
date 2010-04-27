#!/usr/bin/env python

''' class Forest is a collection of nodes, and is responsible for loading/dumping
    the forest.
    The real classes Node and Hyperedge are defined in node_and_hyperedge.py.

	N.B. do not remove sp for an already removed forest.

	Sep 2009: Adapted for ISI translation forest.
'''

# 199 2345
# 1   DT [0-1]	0 
# ...
# 6   NP [0-2]	1 
# 	  1 4 ||| 0=-5.342
# ...

## N.B. node ID might be 123-45-67 where -x are subcats due to annotations.
	
import sys, os, re
import math
import time
import copy

logs = sys.stderr

from utility import getfile, words_to_chars
from tree import Tree

from svector import Vector   # david's pyx, instead of my fvector
from node_and_hyperedge import Node, Hyperedge

from bleu import Bleu
import oracle

from convert_forest import quoteattr
from utility import desymbol

print_merit = False
cache_same = False

base_weights = Vector("lm1=2 gt_prob=1")

def remove_blacklist(fv):
	## evil
	del fv["k-value"]
	del fv["min-top-k"]
	del fv["value"]
	del fv["count"]
	del fv["rootcount"]
	#ulf!
	del fv["sid"]
	del fv["ulf"]
	del fv["u_tb_cost"]
	#new
	del fv["outm1n"]
	del fv["psm1n"]
	
##  	if "outm1n" in fv:
##  		fv["outm1n"] /= 40.
##  		fv["psm1n"] /= 40.
	return fv	


class Forest(object):
	''' a collection of nodes '''

	def copy(self):
		'''must be deep!'''
		return copy.deepcopy(self)
		
	def size(self):
		''' return (num_nodes, num_edges) pair '''
		return len(self.nodes), self.num_edges ##sum([len(node.edges) for node in self.nodes.values()])

	def __init__(self, num, sentence, cased_sent, tag=""):
		self.tag = tag
		self.num = num
		self.nodes = {}  ## id: node
		self.nodeorder = [] #node

		self.sent = sentence
		# a backup of cased, word-based sentence, since sent itself is going to be lowercased and char-based.
		self.cased_sent = cased_sent
		self.len = len(self.sent)
		self.wlen = len(self.cased_sent)

		self.cells = {}   # cells [(2,3)]...
		self.num_edges = 0

		self.weights = base_weights # baseline

	def __len__(self):
		"sentence length"
		return self.len

	def add_node(self, node):
		self.nodes[node.iden] = node
		self.nodeorder.append(node)
		
		node.forest = self ## important backpointer!
		node.wrd_seq = self.sent[node.span[0]: node.span[1]]

	def rehash(self):
		''' after pruning'''

		for i in xrange(self.len+1):
			for j in xrange(i, self.len+1): ## N.B. null span allowed
				self.cells[(i,j)] = []
		
		self.num_edges = 0
		for node in self:
			self.cells[node.span].append(node)
			self.num_edges += len(node.edges)

	def clear_bests(self):
		for node in self:
			node.bestres = None

# 	def adjust_output(self, (sc, tr, fv)):
# 		## subs[0]: remove TOP level
# 		## no longer turning into negative!
# 		return sc, " ".join(tr), fv   #tr.cased_str(self.cased_sent), fv #.subs[0]
	
	def bestparse(self, weights=base_weights, adjust=True):
		self.clear_bests()

		return self.root.bestparse(weights)

	def prep_kbest(self, weights=base_weights):
		self.bestparse(weights)
		for node in self:
			## set up klist and kset, but no computation
			node.prepare_kbest()
			
		return self.root.bestres[0]		

	def iterkbest(self, weights, maxk, threshold):
		''' (lazy) generator '''

		bestscore = self.prep_kbest(weights)
		root = self.root
		for k in xrange(maxk):
			root.lazykbest(k+1)
			if root.fixed or threshold is not None and root.klist[k][0] > bestscore + threshold:
				break			
			else:
				ret = root.klist[k]
				# for psyco
				yield ret
		
	def lazykbest(self, k, weights=base_weights, sentid=0, threshold=None):

		basetime = time.time()

		bestscore = self.prep_kbest(weights)
		
		self.root.lazykbest(k)

		if threshold is not None:
			for i, (sc, tr) in enumerate(self.root.klist):
				if sc > bestscore + threshold:
					self.root.klist = self.root.klist[:i]
					break			

		print >> logs, "sent #%s, %d-best computed in %.2lf secs" % \
			  (self.tag, k, time.time() - basetime)

	@staticmethod
	def load(filename, lower=True, sentid=0):
		'''now returns a generator! use load().next() for singleton.
		   and read the last line as the gold tree -- TODO: optional!
		   and there is an empty line at the end
		'''

		file = getfile(filename)
		line = None
		total_time = 0
		num_sents = 0
		
		while True:			
			
			start_time = time.time()
			##'\tThe complicated language in ...\n"
			## tag is often missing
			line = file.readline()  # emulate seek
			if len(line) == 0:
				break
			try:
				## strict format, no consecutive breaks
# 				if line is None or line == "\n":
# 					line = "\n"
# 					while line == "\n":
# 						line = file.readline()  # emulate seek
						
				tag, sent = line.split("\t")   # foreign sentence
				
			except:
				## no more forests
				yield None
				continue

			num_sents += 1

			# caching the original, word-based, true-case sentence
			sent = sent.split() ## no splitting with " "
			cased_sent = sent [:]			
			if lower:
				sent = [w.lower() for w in sent]   # mark johnson: lowercase all words

			sent = words_to_chars(sent, encode_back=True)  # split to chars

			## read in references
			refnum = int(file.readline().strip())
			refs = []
			for i in xrange(refnum):
				refs.append(file.readline().strip())

			## sizes: number of nodes, number of edges (optional)
			num, nedges = map(int, file.readline().split("\t"))   

			forest = Forest(num, sent, cased_sent, tag)

			forest.refs = refs
			forest.bleu = Bleu(refs=refs)  ## initial (empty test) bleu; used repeatedly later
			
			forest.labelspans = {}
			forest.short_edges = {}
			forest.rules = {}

			for i in xrange(1, num+1):

				## '2\tDT* [0-1]\t1 ||| 1232=2 ...\n'
				## node-based features here: wordedges, greedyheavy, word(1), [word(2)], ...
				line = file.readline()
				try:
					keys, fields = line.split(" ||| ")
				except:
					keys = line
					fields = ""

				iden, labelspan, size = keys.split("\t") ## iden can be non-ints
				size = int(size)

				fvector = Vector(fields) #
				remove_blacklist(fvector)
				node = Node(iden, labelspan, size, fvector, sent)
				forest.add_node(node)

				if cache_same:
					if labelspan in forest.labelspans:
						node.same = forest.labelspans[labelspan]
						node.fvector = node.same.fvector
					else:
						forest.labelspans[labelspan] = node

				for j in xrange(size):
					is_oracle = False

					## '\t1 ||| 0=8.86276 1=2 3\n'
					## N.B.: can't just strip! "\t... ||| ... ||| \n" => 2 fields instead of 3
					tails, rule, fields = file.readline().strip("\t\n").split(" ||| ")

					if tails != "" and tails[0] == "*":  #oracle edge
						is_oracle = True
						tails = tails[1:]

					tails = tails.split() ## N.B.: don't split by " "!
					tailnodes = []
					lhsstr = [] # 123 "thank" 456

					for x in tails:
						if x[0]=='"': # word
							lhsstr.append(desymbol(x[1:-1]))  ## desymbol here and only here; dump will call quoteattr
						else: # variable

							assert x in forest.nodes, "BAD TOPOL ORDER: node #%s is referred to " % x + \
										 "(in a hyperedge of node #%s) before being defined" % iden
							tail = forest.nodes[x]
							tailnodes.append(tail)
							lhsstr.append(tail)

##					use_same = False
##					if fields[-1] == "~":
##						use_same = True
##						fields = fields[:-1]
						
					fvector = Vector(fields) #
					remove_blacklist(fvector)

					edge = Hyperedge(node, tailnodes, fvector, lhsstr)
					edge.rule = rule

					## new
					x = rule.split()
					edge.ruleid = int(x[0])
					if len(x) > 1:
						forest.rules[edge.ruleid] = " ".join(x[1:]) #, None)
						
##					if cache_same:

##						short_edge = edge.shorter()
##						if short_edge in forest.short_edges:
##							edge.same = forest.short_edges[short_edge]
##							if use_same:
##								edge.fvector += edge.same.fvector
##						else:
##							forest.short_edges[short_edge] = edge

					node.add_edge(edge)
					if is_oracle:
						node.oracle_edge = edge
					
				if node.sp_terminal():
					node.word = node.edges[0].subs[0].word

			## splitted nodes 12-3-4 => (12, 3, 4)
			tmp = sorted([(map(int, x.iden.split("-")), x) for x in forest.nodeorder])   
			forest.nodeorder = [x for (_, x) in tmp]

			forest.rehash()
			sentid += 1
			
##			print >> logs, "sent #%d %s, %d words, %d nodes, %d edges, loaded in %.2lf secs" \
##				  % (sentid, forest.tag, forest.len, num, forest.num_edges, time.time() - basetime)

			forest.root = node
			node.set_root(True)

			line = file.readline()

			if line is not None and line.strip() != "":
				if line[0] == "(":
					forest.goldtree = Tree.parse(line.strip(), trunc=True, lower=True)
					line = file.readline()
			else:
				line = None

			total_time += time.time() - start_time

			if num_sents % 100 == 0:
				print >> logs, "... %d sents loaded (%.2lf secs per sent) ..." \
					  % (num_sents, total_time/num_sents)
				
			yield forest

		# better check here instead of zero-division exception
		if num_sents == 0:
			print >> logs, "No Forests Found! (empty input file?)"
			yield None # new: don't halt
		
		Forest.load_time = total_time
		print >> logs, "%d forests loaded in %.2lf secs (avg %.2lf per sent)" \
			  % (num_sents, total_time, total_time/(num_sents+0.001))

	@staticmethod
	def loadall(filename):
		forests = []
		for forest in Forest.load(filename):
			forests.append(forest)
		return forests

	def dump(self, out=sys.stdout):
		'''output to stdout'''
		# wsj_00.00       No , it was n't Black Monday .
		# 199
		# 1	DT [0-1]	0 ||| 12321=... 46456=...
		# ...
		# 6	NP [0-2]	1 ||| 21213=... 7987=...
		# 	1 4 ||| 0=-5.342
		# ...

		if type(out) is str:
			out = open(out, "wt")

		# CAUTION! use original cased_sent!
		print >> out, "%s\t%s" % (self.tag, " ".join(self.cased_sent))
		print >> out, len(self.refs)
		for ref in self.refs:
			print >> out, ref
		
		print >> out, "%d\t%d" % self.size()  # nums of nodes and edges
		for node in self:

			oracle_edge = node.oracle_edge if hasattr(node, "oracle_edge") else None
			
			print >> out, "%s\t%d |||" % (node.labelspan(separator="\t"), len(node.edges)),
			if hasattr(node, "same"):
				print >> out, " "
			else:
				print >> out, node.fvector
				
			##print >> out, "||| %.4lf" % node.merit if print_merit else ""

			rulecache = set()
			for edge in node.edges:

				is_oracle = "*" if (edge is oracle_edge) else ""

				## caution: pruning might change caching, so make sure rule is defined in the output forest
				if edge.ruleid in rulecache:
					rule_print = str(edge.ruleid)
				else:
					rule_print = "%s %s" % (edge.ruleid, self.rules[edge.ruleid])
					rulecache.add(edge.ruleid)

				tailstr = " ".join([quoteattr(x) if type(x) is str else x.iden for x in edge.lhsstr])
				print >> out, "\t%s%s ||| %s ||| %s" \
							% (is_oracle, tailstr, rule_print, edge.fvector)
					 
		print >> out  ## last blank line

	def __iter__(self):
		for node in self.nodeorder:
			yield node

	def reverse(self):
		for i in range(len(self.nodeorder)):
			ret = self.nodeorder[-(i+1)]			
			yield ret

	## from oracle.py
	def recover_oracle(self):
		'''oracle is already stored implicitly in the forest
		returns best_score, best_parseval, best_tree, edgelist
		'''
		edgelist = self.root.get_oracle_edgelist()
		fv = Hyperedge.deriv2fvector(edgelist)
		tr = Hyperedge.deriv2tree(edgelist)
		bleu_p1 = self.bleu.rescore(tr)
		return bleu_p1, tr, fv, edgelist


	def compute_oracle(self, weights, model_weight=0, bleu_weight=1, store_oracle=False):
		'''idea: annotate each hyperedge with oracle state. and compute logBLEU for
		each node. NOTE that BLEU is highly non-decomposable, so this dynamic programming
		is a very crude approximation. alternatively, we can use Tromble et al decomposable
		BLEU (so that it becomes viterbi deriv), but then we need to tune the coefficients.'''


		basetime = time.time()
		bleu, hyp, fv, edgelist = self.root.compute_oracle(weights, self.bleu, \
																											 self.len, self.wlen, \
																											 model_weight, bleu_weight)
		bleu = self.bleu.rescore(hyp) ## for safety, 755 bug
		
##		fv2 = Hyperedge.deriv2fvector(edgelist)
##		assert str(fv) == str(fv2), "\n%s\n%s" % (fv, fv2)
##		hyp = " ".join(hyp)
##		print >> logs, "my oracle computed in %.2lf secs" % (time.time() - basetime)

		if store_oracle:
			for edge in edgelist:
				edge.head.oracle_edge = edge
				
		return bleu, hyp, fv, edgelist

def get_weights(w):
	'''input: either filename or weightstr.'''

	#TODO: modify svector (exception handling, robustness)

	if w.strip() == "":
		weights = Vector()
	else:
		if not (w.find(":") >= 0 or w.find("=") >= 0):
			w = open(w).readline().strip()

		# now outside
#		w = w.replace(":", "=").replace(",", " ") 
		weights = Vector(w)

	print >> logs, 'using weights: "%s...%s" (%d fields)' \
				% (w[:10], w[-10:], len(weights))

	return weights

if __name__ == "__main__":

	try:
		import psyco
		psyco.full()
	except:
		pass
	
	import optparse
	optparser = optparse.OptionParser(usage="usage: cat <forests> | %prog [options (-h for details)]")
	## default value for k is different for the two modes: fixed-k-best or \inf-best
	optparser.add_option("-k", "", dest="k", type=int, help="k-best", metavar="K", default=None)
	optparser.add_option("", "--thres", dest="threshold", type=float, \
						 help="threshold/margin", metavar="THRESHOLD", default=None)
	optparser.add_option("", "--inf", dest="infinite", action="store_true", help="\inf-best", default=False)
	optparser.add_option("", "--dump", dest="dump", action="store_true", help="\inf-best", default=False)
	optparser.add_option("", "--id", dest="sentid", type=int, help="sentence id", metavar="ID", default=0)
	optparser.add_option("-R", "--range", dest="first", type=str, \
						 help="dump forests from F to T (inclusive) to stdout", metavar="F:T", default=None)
	## weights
	optparser.add_option("-w", "--weights", dest="weights", type=str, help="weights file or str", metavar="WEIGHTS", default="lm1=2 gt_prob=1")
	optparser.add_option("", "--oracle", dest="compute_oracle", action="store_true", help="compute oracles", default=False)
	optparser.add_option("", "--fear", dest="compute_fear", action="store_true", help="compute fears", default=False)
	optparser.add_option("", "--hope", dest="hope", type=float, help="hope weight", default=0)
	optparser.add_option("", "--recover", dest="recover_oracle", action="store_true", help="recover oracles", default=False)
	
	(opts, args) = optparser.parse_args()

	if opts.first is not None:
		first, last = map(int, opts.first.split(":"))

	weights = get_weights(opts.weights)

	davidoraclebleus = Bleu()
	myoraclebleus = Bleu()
	myfearbleus = Bleu()
	davidscores = 0
	myscores = 0
	myfearscores = 0
	onebestscores = 0
	onebestbleus = Bleu()
	

	for i, f in enumerate(Forest.load("-")):

		if f.tag == "1":
			f.tag = "sent.%d" % (i+1)

		if opts.first is not None:
			if i+1 >= first:
				f.dump()
			if i+1 >= last:
				break
			continue
		elif opts.dump:
			f.dump()

		if not opts.infinite:
			if opts.k is None:
				opts.k = 1

			f.lazykbest(opts.k, weights=weights, sentid=f.tag, threshold=opts.threshold)
			print >> logs, "%d\t%s" % (len(f.root.klist), f.tag)			

			for k, res in enumerate(f.root.klist):
				score, hyp, fv = res
				hyp = (hyp)
				hyp_bleu = f.bleu.rescore(hyp)
				print >> logs, "k=%d\tscore=%.4lf\tbleu+1=%.4lf\tlenratio=%.2lf\n%s" % (k+1, score, hyp_bleu, f.bleu.ratio(), hyp)
				if k == 0:
					onebestscores += score
					onebestbleus += f.bleu.copy()

			if opts.recover_oracle:
				oracle_bleu, oracle_hyp, oracle_fv = f.recover_oracle()[:3]
				oracle_score = oracle_fv.dot(weights)
				oracle_hyp = (oracle_hyp)
				davidoraclebleus += f.bleu.copy()
				davidscores += oracle_score
				print >> logs,  "oracle\tscore=%.4lf\tbleu+1=%.4lf\tlenratio=%.2lf\n%s" % \
							(oracle_score, oracle_bleu, f.bleu.ratio(), oracle_hyp)			

			if opts.compute_oracle:
				bleu, hyp, fv, edgelist = f.compute_oracle(weights, opts.hope, 1)
				bleu = f.bleu.rescore(hyp)
				mscore = weights.dot(fv)
				print  >> logs, "moracle\tscore=%.4lf\tbleu+1=%.4lf\tlenratio=%.2lf\n%s" % \
							(mscore, f.bleu.fscore(), f.bleu.ratio(), hyp)
				
				myoraclebleus += f.bleu.copy()
				myscores += mscore

			if opts.compute_fear:
				bleu, hyp, fv, edgelist = f.compute_oracle(weights, 1, -1)
				bleu = f.bleu.rescore(hyp)
				mscore = weights.dot(fv)
				print  >> logs, "   fear\tscore=%.4lf\tbleu+1=%.4lf\tlenratio=%.2lf\n%s" % \
							(mscore, f.bleu.fscore(), f.bleu.ratio(), hyp)
				
				myfearbleus += f.bleu.copy()
				myfearscores += mscore

		else:
			if opts.k is None:
				opts.k = 100000 ## inf
			for res in f.iterkbest(opts.k, threshold=opts.threshold):
				print >> logs,  "%.4lf\n%s" % (f.adjust_output(res)[:2])

		if i % 10 == 9:
			print >> logs,  "overall 1-best deriv bleu = %.4lf (%.2lf) score = %.4lf" \
						% (onebestbleus.score_ratio() + (onebestscores/(i+1),))
			if opts.recover_oracle:
				print >> logs,  "overall david oracle bleu = %.4lf (%.2lf) score = %.4lf" \
							% (davidoraclebleus.score_ratio() + (davidscores/(i+1),))
			if opts.compute_oracle:
				print >> logs,  "overall my    oracle bleu = %.4lf (%.2lf) score = %.4lf" \
							% (myoraclebleus.score_ratio() + (myscores/(i+1),))

			if opts.compute_oracle:
				print >> logs,  "overall my      fear bleu = %.4lf (%.2lf) score = %.4lf" \
				      % (myfearbleus.score_ratio() + (myfearscores/(i+1),))


	if opts.compute_oracle:
		print >> logs,  "overall 1-best deriv bleu = %.4lf (%.2lf) score = %.4lf" \
		      % (onebestbleus.score_ratio() + (onebestscores/(i+1),))
		print >> logs,  "overall my    oracle bleu = %.4lf (%.2lf) score = %.4lf" \
		      % (myoraclebleus.score_ratio() + (myscores/(i+1),))

