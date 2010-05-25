#!/usr/bin/env python

import sys

class DottedRule(object):

    __slots__ = "edge", "dot"

    def __init__(self, edge, dot=0):
        self.edge = edge
        self.dot = dot

    def tree_size(self):
        return self.edge.rule.tree_size() # number of non-variable nodes in the lhs tree

    def advance(self):
        '''advance the dot by one position (in-place!)'''
        self.dot += 1
        return self

    def next_symbol(self):
        return self.edge.lhsstr[self.dot]

    def end_of_rule(self):
        return self.dot == len(self.edge.lhsstr)

    def __eq__(self, other):
        # TODO: only compare those after dot
        return self.edge == other.edge and self.dot == other.dot
    
class LMState(object):

    ''' stack is a list of dotted rules(hyperedges, dot_position) '''
    
    __slots__ = "stack", "_trans", "score", "step"

    order = 3  # n-gram order
    lm = None
    vocab = None
    weights = None

    @staticmethod
    def set_lm(lm, vocab, weights, order=3):
        LMState.lm = lm
        LMState.vocab = vocab
        LMState.weights = weights
        LMState.lm_weight = weights.lm_weight
        LMState.order = order

    def __init__(self, stack, trans, score=0, step=0):
        self.stack = stack
        self._trans = trans
        self.score = score
        self.step = step
        self.scan()

    def predict(self):
        next_symbol = self.next_symbol()
        if type(next_symbol) is Node:
            for edge in next_symbol.edges:
                yield LMState(self.stack + [DottedRule(edge)], 
                              self._trans, 
                              self.score + edge.fvector.dot(LMState.weights),
                              self.step + self.stack[-1].tree_size())

    def lmstr(self):
        # TODO: cache real lmstr
        return self._trans[-LMState.order+1:]

    def scan(self):
        while True:
            symbol = self.next_symbol()
            if type(symbol) is str:
                this = LMState.vocab.index(symbol)
                self.stack[-1].advance() # dot ++
                self.score += LMState.lm.wordprob(this, self.lmstr()) * lm_weight
                self._trans += [this]
            else:
                break

    next_symbol = lambda self: self.stack[-1].next_symbol
    end_of_rule = lambda self: self.stack[-1].end_of_rule

    def complete(self):
        if self.end_of_rule():
            yield LMState(self.stack[:-2] + [self.stack[-2].advance()], 
                          self._trans,
                          self.score, 
                          self.step + self.stack[-1].tree_size())

    def __eq__(self, other):
        ## calls DottedRule.__eq__()
        return self.stack == other.stack and \
               self.lmstr() == other.lmstr()

    def is_final(self):
        ''' a complete translation'''
        # TOP' -> <s> TOP </s> . (dot at the end)
        return len(self.stack) == 1 and self.stack[0].end_of_rule()

    def trans(self):
        '''recover translation from lmstr'''
        return 
