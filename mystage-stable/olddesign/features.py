#!/usr/bin/env python

''' Feature -> FeatureClasses -> FeatureInstances
'''

import sys

from utility import quantize, make_punc

class Feature(object):

	__slots__ = "Id", "rstr"
	
	def __init__(self, id=-1):
		''' -1 means not in feature set'''		
		pass
	
	def __str__(self):
		''' like (RB DT _ ADJP)'''
		if self.rstr is None:
			self.rstr = self.dostr()
		return self.rstr

	@staticmethod
	def nametag():
		''' like Rule:0:0:0:0:0:0:0:1'''
		pass

	def fullstr(self):
		return self.__str__() + " " + self.nametag()

	@staticmethod
	def count(tree):
		'''count the counts of features of this class on this (sub-) tree'''
		pass

class NodeFeature(Feature):
	'''Heavy, WordEdges'''
	pass
class HyperedgeFeature(Feature):
	'''Rule, NGram'''
	pass

class Heavy(NodeFeature):
	## 425858  Heavy ((5 4) (PP '' _))
	## (binned_len binned_distance_to_end) (label final_punc following_punc)
	## note: puncs mean words, not POS tags

	__slots__ = "binned_len", "distance", "label", "final_punc", "follow_punc"
	
	@staticmethod
	def nametag():
		return "Heavy"

	def __init__(self, binned_len, distance, label, final_punc, follow_punc):
				
		self.dostr()

	def dostr(self):
		return "((%d %d) (%s %s %s))" % (self.binned_len, self.distance, self.label, self.final_punc, self.follow_punc)
	
	@staticmethod
	def count(tree, sentence):
		if tree.is_terminal():
			return None
		
		binned_len = tree.binned_span_width())
		distance = quantize(len(sentence) - tree.span[1])   ## will be moved into tree

		## will be integrated into tree by passing a sentence, but not storing the sentence
		final_punc = make_punc(sentence[tree.span[1] - 1])   
		follow_punc = make_punc(sentence[tree.span[1]])

		return Heavy(binned_len, distance, tree.label, final_punc, follow_punc)	
		

class Rule(HyperedgeFeature):
	pass
