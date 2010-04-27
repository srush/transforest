#!/usr/bin/env python

''' PARSEVAL. has a switch of removing punctuations.
    KNOWN bugs: unary cycles like (NP (NP ...)) will be treated as (NP ...)
'''

import sys
import math
import copy
from tree import *
from utility import xzip

logs = sys.stderr

debug = True

evalb_delete = set([",", ":", "``", "''", "."])

class OPT(object):
	pass

opts = OPT()
opts.debug = False
opts.backoff = True
opts.evalb = False

def is_evalb_punc(p):
	''' these puncs are to be deleted by standard EVALB (COLLINS.prm) '''

	return p in evalb_delete

def merge_labels(lst, mapping=lambda x:x):
	'''mapping is the index-mapping due to punc-removing.'''
	return [merge_label(x, mapping) for x in lst]

def merge_label((label, (a, b)), mapping=lambda x:x):

	return (("ADVP" if label == "PRT" else label), (mapping(a), mapping(b)))

def check_puncs(pos1, pos2):

	'''TODO: cache the mapping from pos2, and just check pos1 with pos2'''
	
	assert len(pos1) == len(pos2), "different sentence lengths!\n%s\n%s" % (str(test_tree), str(gold_tree))

	idx_mapping = {}
	j = 0
	last_is_punc = True
	for i, (a, b) in enumerate(zip(pos1, pos2)):
		if is_evalb_punc(a) ^ is_evalb_punc(b):
			## bad matching!
			return None

		if not last_is_punc:
			j += 1
		idx_mapping [i] = j
		
		last_is_punc = is_evalb_punc(a)

	if not last_is_punc:
		j += 1
	idx_mapping[i+1] = j

	return lambda x: idx_mapping[x]


def complete_match(m, t, g):
	return int(m == t == g)

class Parseval(object):

	__slots__ = "testbr", "goldbr", "matchbr", "cb", "cm", "PR", "RE", "F", "num_sent", "_changed", "complete"

	goldtree = None

	@staticmethod
	def parseval_spans(goldtree):
		idx_mapping = check_puncs(goldtree.tag_seq, goldtree.tag_seq)
		return merge_labels(goldtree.all_label_spans()[1:], idx_mapping)   # omit TOP
	
##		return Parseval.gold_brackets
##	@staticmethod
##	def new_batch(goldtree):
##		''' start a new batch, for nbest trees sharing the same gold tree.'''

##		Parseval.goldtree = goldtree
##		Parseval.idx_mapping = check_puncs(goldtree.tag_seq, goldtree.tag_seq)
##		Parseval.gold_brackets = merge_labels(goldtree.all_label_spans()[1:], idx_mapping)   # omit TOP
##		return Parseval.gold_brackets

##	@staticmethod
##	def in_batch(testtree):
##		'''another example in the current batch'''
##		assert Parseval.goldtree is not None
		
	@staticmethod
	def get_parseval(matched, test, gold):
		a = Parseval()
		a.matchbr = matched + 0.0
		a.testbr = test
		a.goldbr = gold
		a.complete = complete_match(matched, test, gold)
		a.num_sent = 1
		a._changed = True
		return a

	def __init__(self, test=None, gold=None, crossing=False, del_puncs=True):
		''' return a Parseval object, containing all the info'''

		# identity-mapping
		
		if test is None:
			''' by default is empty'''
			self.testbr = self.goldbr = self.matchbr = 0
			self.num_sent = 0
			self.complete = 0
			self._changed = True
			return			

		## N.B. this implementation is incorrect when there is duplicate unary labels
		## like (NP (NP ...)), in which case the parseval returned might be lower

		if type(test) is str:
			test = Tree.parse(test)

		if type(gold) is str:
			gold = Tree.parse(gold)

		sent = test.get_sent()
		assert sent == gold.get_sent(), "sentence mismatch!\n%s %s \n%s %s" % \
			   (" ".join(sent), test, " ".join(gold.get_sent()), gold) 

		if del_puncs:
			idx_mapping = check_puncs(test.tag_seq, gold.tag_seq)

			if idx_mapping is None:    ## punc mismatch ==> make it zero
				## VERY CAREFUL HERE! setting all zeros neglects this error example (as in standard evalb)
				## but do not set it to 0.0 otherwise division error (over-optimistic)
				## setting only matchbr to zero does not neglect this example, but over-pessimistic
				## johnson's program computes everything up to the error point (but still not perfect)

				if not opts.backoff:
					if opts.evalb:
						self.testbr = 0
						self.goldbr = 0
					else:
						self.testbr = len(test.all_label_spans()) - 1
						self.goldbr = len(gold.all_label_spans()) - 1

					self.matchbr = 0.0
					self.complete = 0
					self.num_sent = 1
					self._changed = True				
					return
				else:
					## resort to vanilla evalb -- no punc deleting
					idx_mapping = lambda x:x   ## add back the puncs				
				
		else:
			idx_mapping = lambda x:x #dict([(x, x) for x in range(len(tree.tag_seq))])
			
		test_brackets = merge_labels(test.all_label_spans()[1:], idx_mapping)   # omit TOP
		gold_brackets = merge_labels(gold.all_label_spans()[1:], idx_mapping)

		matched = set(test_brackets) & set(gold_brackets)  ## NOT "and"!!

		if opts.debug:
			print >> logs, "sentence=", " ".join(map(str, enumerate(sent)))
			print >> logs, "idx mapping=", \
				  ", ".join(["%d->%d" % (a, idx_mapping(a)) for a in range(len(sent)+1)])
			print >> logs, "matched brackets=", " ".join(map(str, matched))

# 		print >> logs, "test = ", test_brackets
# 		print >> logs, "gold = ", gold_brackets
# 		print >> logs, "matched", matched		

		self.testbr = len(test_brackets)
		self.goldbr = len(gold_brackets)
		self.matchbr = len(matched) + .0

		self.complete = complete_match(self.matchbr, self.testbr, self.goldbr)

		self.num_sent = 1

		self._changed = True
		
		if crossing:
			## compute crossing brackets
			pass
		

	def recompute(self):
		if self.matchbr == 0:
			self.PR = self.RE = self.F = 0
		else:
			self.PR = self.matchbr / self.testbr
			self.RE = self.matchbr / self.goldbr
			self.F = 2 * self.matchbr / (self.testbr + self.goldbr)

		##self.complete = (self.matchbr == self.testbr == self.goldbr)

		self._changed = False

##		self.cm = (self.testbr == self.matchbr) and (self.goldbr == self.matchbr)

	def brs(self):
		return (self.goldbr, self.testbr, self.matchbr)
		
	def __str__(self):

		if self._changed:
			self.recompute()
			
		return "%d sents, prec %d/%d %.4lf\trecall %d/%d %.4lf\tf-score %.4lf complete %.3lf" \
			   % (self.num_sent, \
				  self.matchbr, self.testbr, self.PR, \
				  self.matchbr, self.goldbr, self.RE, \
				  self.F, self.complete / float(self.num_sent))

	def __iadd__(self, other):
		''' in-place add! has to return self '''

		self.testbr += other.testbr
		self.goldbr += other.goldbr
		self.matchbr += other.matchbr

		self.num_sent += other.num_sent

		self.complete += other.complete
		self._changed = True

		return self

	def fscore(self):
		if self._changed:
			self.recompute()
		return self.F

	def __cmp__(self, other):

		d = self.fscore() - other.fscore()
		if math.fabs(d) < 1e-6:
			return 0
		return 1 if d<0 else -1

	def copy(self):
		''' return a copy of self'''
		return copy.copy(self)
	

if __name__ == "__main__":

##	a = ["DT", ".", "A", "B", ",", "''", "C"]
##	b = ["nn", ".", "A", "C", ",", "''", "E"]

##	print check_puncs(a, b)

	import optparse
	optparser = optparse.OptionParser(usage="usage: %prog [options (-h)] <1-best parses> <goldtrees>")
	optparser.add_option("-f", "--first", dest="first", type="int", \
						 help="first NUM sentences only", metavar="NUM", default=None)
	optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", \
						 help="trace for each sentence", default=False)
	optparser.add_option("-d", "--debug", dest="debug", action="store_true", \
						 help="debug info: matched brackets", default=False)
	optparser.add_option("-E", "--EVALB", dest="evalb", action="store_true", \
						 help="exact EVALB mode: neglecting error sentences due to puncs", default=False)
	optparser.add_option("-p", "--punc", dest="punc", action="store_false", \
						 help="keep punctuations (default is to delete them first)", default=True)
	optparser.add_option("-b", "--nobackoff", dest="backoff", action="store_false", \
						 help="no backing off to vanilla evalb", default=True)
	optparser.add_option("-H", "--histogram", dest="histogram", type="int", \
						 help="print histogram according to binned length", metavar="BIN_LEN")


	(opts, args) = optparser.parse_args()

	if opts.histogram:
		histo = {}
			
	from readkbest import readonebest
	testtrees = readonebest(args[0])
	goldtrees = readonebest(args[1])

	parseval = Parseval()
	for i, (ta, tb) in enumerate(xzip(testtrees, goldtrees)): 

		if opts.first is not None and i >= opts.first:
			break

		par = Parseval(ta, tb, del_puncs=opts.punc)
		if opts.verbose:
			print "%d\t%s" % (i+1, par)
		parseval += par

		if opts.histogram:
			lentr = len(ta) / opts.histogram  + 1
			if lentr not in histo:
				histo[lentr] = Parseval()
			histo[lentr] += par

	print parseval

	if opts.histogram:
		for i in histo:
			print i * opts.histogram, histo[i].fscore()
								 

