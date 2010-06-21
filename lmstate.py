#!/usr/bin/env python

import sys
from node_and_hyperedge import Hyperedge, Node
from svector import Vector
from rule import Rule

import gflags as flags
FLAGS=flags.FLAGS

flags.DEFINE_boolean("dp", False, "dynamic programming")    
flags.DEFINE_boolean("complete", False, "use complete step")    

class DottedRule(object):

    __slots__ = "edge", "dot", "_hash"

    def __init__(self, edge, dot=0):
        self.edge = edge
        self.dot = dot
        self.rehash()

    def rehash(self):
        self._hash = hash((self.edge, self.dot))

    def tree_size(self):
        return self.edge.rule.tree_size() # number of non-variable nodes in the lhs tree

    def advance(self):
        '''advance the dot by one position (in-place!)'''
        self.dot += 1
        self.rehash()

    def advanced(self):
        '''advance the dot by one position (new!)'''
        return DottedRule(self.edge, self.dot+1)

    def next_symbol(self):
        try:
            return self.edge.lhsstr[self.dot] # Node or str
        except:
            print self.edge.lhsstr, self.dot
            assert False

    def end_of_rule(self):
        return self.dot == len(self.edge.lhsstr)

    def __eq__(self, other):
        # TODO: only compare those after dot
        return self.edge == other.edge and self.dot == other.dot

    def __str__(self):
        return self.edge.dotted_str(self.dot)

    def __hash__(self):
        return self._hash
    
class LMState(object):

    ''' stack is a list of dotted rules(hyperedges, dot_position) '''
    
    __slots__ = "stack", "_trans", "score", "step", "fvector", "_hash"

    weights = None
    lm = None
    dp = False

    @staticmethod
    def init(lm, weights, dp=None):
        LMState.lm = lm
        LMState.weights = weights
        if dp is None:
            dp = FLAGS.dp
        LMState.dp = dp

    @staticmethod
    def start_state(root):
        ''' None -> <s>^{g-1} . TOP </s> '''
        lmstr = LMState.lm.raw_startsyms()
        lhsstr = lmstr + [root] + LMState.lm.raw_stopsyms()
        edge = Hyperedge(None, [root], Vector(), lhsstr)
        edge.rule = Rule.parse("ROOT(TOP) -> x0 ### ")        
        return LMState([DottedRule(edge, dot=len(lmstr))], LMState.lm.startsyms(), 0, 0)

    def __init__(self, stack, trans, score=0, step=0, fvector=Vector()):
        self.stack = stack
        self._trans = trans
        self.score = score
        self.step = step
        self.fvector = fvector
        self.scan()        

    def predict(self):
        if not self.end_of_rule():
            next_symbol = self.next_symbol()
            if type(next_symbol) is Node:
                for edge in next_symbol.edges:
                    # N.B.: copy trans
                    yield LMState(self.stack + [DottedRule(edge)], 
                                  self._trans[:], 
                                  self.score + edge.fvector.dot(LMState.weights),
                                  self.step + edge.rule.tree_size(),
                                  self.fvector + edge.fvector)

    def lmstr(self):
        # TODO: cache real lmstr
        return self._trans[-LMState.lm.order+1:] if LMState.dp else self._trans

    def scan(self):

        while not self.is_final():
            # scan
            if not self.end_of_rule():                
                symbol = self.next_symbol()
                if type(symbol) is str:
                    # TODO: cache lm index
                    this = LMState.lm.word2index(symbol)
                    self.stack[-1].advance() # dot ++
                    #TODO fix ngram
                    lmscore = LMState.lm.ngram.wordprob(this, self.lmstr())
                    self.score += lmscore * LMState.weights.lm_weight
                    self.fvector["lm"] += lmscore
                    self._trans += [this,]
                else:
                    break
            else:
                if FLAGS.complete:
                    break
                else: # pop stack
                    self.step += self.stack[-1].tree_size()
                    self.stack = self.stack[:-2] + [self.stack[-2].advanced()]

        self.rehash()

    next_symbol = lambda self: self.stack[-1].next_symbol()
    end_of_rule = lambda self: self.stack[-1].end_of_rule()

    def complete(self):
        if self.end_of_rule():
            # N.B.: copy trans
            yield LMState(self.stack[:-2] + [self.stack[-2].advanced()], 
                          self._trans[:],
                          self.score, 
                          self.step + self.stack[-1].tree_size(),
                          Vector(self.fvector)) # copy.copy is slow

    def rehash(self):
        self._hash = hash(tuple(self.stack) + tuple(self.lmstr()))# + (self.score, self.step))

    def __eq__(self, other):
        ## calls DottedRule.__eq__()
        return self.stack == other.stack and \
               self.lmstr() == other.lmstr()

    def __cmp__(self, other):
        return cmp(self.score, other.score)

    def is_final(self):
        ''' a complete translation'''
        # TOP' -> <s> TOP </s> . (dot at the end)
        return len(self.stack) == 1 and self.stack[0].end_of_rule()

    def trans(self):
        '''recover translation from lmstr'''
        return LMState.lm.ppqstr(self._trans[LMState.lm.order-1:-LMState.lm.order+1])

    def __str__(self):
        return "LMState step=%d, score=%.2lf, trans=%s, stack=[%s]" % \
               (self.step, self.score, self.trans(), ", ".join("(%s)" % x for x in self.stack))

    def __hash__(self):
        return self._hash
