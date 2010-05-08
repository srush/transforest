#!/usr/bin/env python

from collections import defaultdict
import sys
import time
logs = sys.stderr

from utility import quoteattr

class Rule(object):

    __slots__ = "lhs", "rhs", "fields", "ruleid"

    def __init__(self, lhs, rhs, fields):
        self.lhs = lhs
        self.rhs = rhs
        self.fields = fields        

    @staticmethod
    def parse(line):
        '''S -> NP VP ### gt_prob=-5 text-lenght=0 plhs=-2'''
        try:
            rule, fields = line.split(" ### ")
            lhs, rhs = rule.strip().split(" -> ")
            rhs = rhs.split()
            return Rule(lhs, rhs, fields)
        except:
            print >> logs, "BAD RULE: %s" % line.strip()
            return None

    def __str__(self):
        return "%s ### %s" % (repr(self), self.fields)

    def __repr__(self):
        return "%s -> %s" % (self.lhs, \
                             " ".join(quoteattr(s[1:-1]) if s[0] == '"' else s \
                                      for s in self.rhs))
    
class RuleSet(defaultdict):
    ''' map from lhs to a list of Rule '''

    def __init__(self, rulefilename):
        '''read in rules from a file'''
        self.ruleid = 0
        defaultdict.__init__(self, list) # N.B. superclass
        print >> logs, "reading rules from %s ..." % rulefilename
        otime = time.time()
        bad = 0
        for i, line in enumerate(open(rulefilename), 1):
            rule = Rule.parse(line)
            if rule is not None:
                self.add_rule(rule)
            else:
                bad += 1
            if i % 100000 == 0:
                print >> logs, "%d rules read (%d BAD)..." % (i, bad) 
            
        print >> logs, "\ntotal %d rules (%d BAD) read in %d secs" % (i, bad, time.time() - otime)

    def add_rule(self, rule):
        self[rule.lhs].append(rule)
        rule.ruleid = self.ruleid
        self.ruleid += 1
 
    def rule_num(self):
        return self.ruleid
