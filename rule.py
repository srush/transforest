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
        '''S(NP("...") VP) -> x1 "..." x0 ### gt_prob=-5 text-lenght=0 plhs=-2'''
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
    def tree_size(self):
        '''number of non-variable nodes in lhs tree'''
        # TODO: -LRB--
        return self.lhs.count("(") - self.lhs.count("\"(")
    
class RuleSet(defaultdict):
    ''' map from lhs to a list of Rule '''

    def __init__(self, rulefilename):
        '''read in rules from a file'''
        defaultdict.__init__(self, list) # N.B. superclass
        self.ruleid = 0
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

if __name__ == "__main__":
    import optparse
    optparser = optparse.OptionParser(usage="usage: rule.py 1>filtered.rules 2>bad.rules")
    
    print >> logs, "start to filting rule set ..."
    otime = time.time()
    bad1 = 0  # ratio > 3
    bad2 = 0  # best 100
    bad = 0   # rhs == null
    filteredrs = defaultdict(list)
    for i, line in enumerate(sys.stdin, 1):
        rule = Rule.parse(line)
        if rule is not None:
            ratioce = float(len(rule.lhs.split()))/float(len(rule.rhs))
            ratioec = float(len(rule.rhs))/float(len(rule.lhs.split()))
            if ratioec > 6 or ratioce > 4:
                bad1 += 1
                print >> logs, "Bad Ratio: %s" % line
            else:
                filteredrs[rule.lhs].append(rule)
        else:
            bad += 1

    for (lhs, rules) in filteredrs.iteritems():
        bad2 += ((len(rules) - 100) if len(rules)>100 else 0)
        rules = rules[:100]
        for rule in rules:
            print "%s" % str(rule)
    print >> logs, "\ntotal %d rules (%d rhs=null; %d ratio>3; %d rhs>100)filtered in %.2lf secs" % (i, bad, bad1, bad2, time.time() - otime)
