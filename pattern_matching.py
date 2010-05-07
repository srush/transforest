#!/usr/bin/env python

import sys
from collections import defaultdict
import time

from node_and_hyperedge import Node, Hyperedge
from rule import Rule, RuleSet

from svector import Vector

from utility import quoteattr, desymbol

logs = sys.stderr

class PatternMatching(object):

    def __init__(self, forest, ruleset, filtered_ruleset, deffields, filterR=False):
        self.forest = forest
        self.ruleset = ruleset
        self.filtered_ruleset = filtered_ruleset
        self.deffields = deffields
        self.filterR = filterR
    
    def lexFrag(self, node):
        '''add lexical frag (lhs, rsh, height) = (PU("."), [], 1)'''
        return ('%s("%s")' % (node.label, node.word), [], 1)
    
    def defFrag(self, node):
        '''# add default frag (lhs, rsh, height) = (PU, [node], 0)'''
        return (node.label, [node], 0)
    
    def addFilterRules(self, fRuleSet, lhs, rules):
        if lhs not in fRuleSet:
            fRuleSet[lhs] = rules
    
    def convert(self):
        '''convert parse forest into translation forest'''
        forest = self.forest
        
        for node in forest:
            
            if node.is_terminal():  # a leaf node
                #append the default frag (lhs, rhs, height) = (PU, node, 0)
                node.frags.append(self.defFrag(node))
                #appedn the lexical frag (lhs, rhs, height) = (PU("."), [], 1)
                frag = self.lexFrag(node)
                node.frags.append(frag)
                # add lexical translation hyperedges
                self.addLexTH(frag[0], node)
                
            else:  # it's a non-terminal node
                #append the default frag (lhs, rhs, height) = (PU, node, 0)
                node.frags.append(self.defFrag(node))
                # add non-terminal translation hyperedges
                self.addNonterHG(node)
                
        return forest
    
    @staticmethod
    def combinetwofrags(basefrag, varfrag, id, lastchild):
        blhs, brhs, bheight = basefrag
        vlhs, vrhs, vheight = varfrag
        height = bheight if bheight > (vheight + 1) else (vheight+1)
        if height >3:
            return None
        
        lhs = "%s %s" % (blhs, vlhs) if id>0 else "%s%s" % (blhs, vlhs)
        if id == lastchild:
            lhs += ")"
        
        rhs = []
        rhs.extend(brhs)
        rhs.extend(vrhs)
       
        return (lhs, rhs, height)

    def addLexTH(self, lhs, node):
        # add lexical translation rules
        ruleset = self.ruleset
        
        if lhs in ruleset:
            rules = ruleset[lhs]

            # add rules to filtered_ruleset
            if self.filterR:
                self.addFilterRules(self.filtered_ruleset, lhs, rules)

            # add all translation hyperedges
            for rule in rules:
                tfedge = Hyperedge(node, [], Vector(rule.fields),\
                                  [desymbol(x[1:-1]) for x in rule.rhs])
                tfedge.rule = rule
                node.tfedges.append(tfedge)
                        
        else: # add a default translation hyperedge (monotonic)        
            rule = Rule(lhs, ['"%s"' % node.word], self.deffields)
            tfedge = Hyperedge(node, [], Vector(self.deffields), \
                                       [node.word])
            tfedge.rule = rule
            ruleset.add_rule(rule)
            node.tfedges.append(tfedge)
        
    def addNonterHG(self, node):
        ruleset = self.ruleset
        
        for edge in node.edges:
            # enumerate all the possible frags
            basefrags = [("%s(" % node.label, [], 1)]
            lastchild = len(edge.subs) - 1
            for (id, sub) in enumerate(edge.subs):
                oldfrags = basefrags
                # cross-product
                basefrags = [PatternMatching.combinetwofrags(oldfrag, frag, id, lastchild) \
                                     for oldfrag in oldfrags for frag in sub.frags]

            # for each frag add translation hyperedges
            for extfrag in basefrags:
                extlhs, extrhs, extheight = extfrag
                # add frags
                if extheight <= 2:
                    node.frags.append(extfrag)
                        
                # add translation hyperedges
                if extlhs in ruleset:
                    rules = ruleset[extlhs]
                            
                    # add rules to filtered_ruleset
                    if self.filterR:
                        self.addFilterRules(self.filtered_ruleset, extlhs, rules)

                    # add all translation hyperedges
                    for rule in rules:
                        rhsstr = [desymbol(x[1:-1]) if x[0]=='"' \
                                          else extrhs[int(x.split('x')[1])] \
                                          for x in rule.rhs]
                        tfedge = Hyperedge(node, extrhs,\
                                                   Vector(rule.fields), rhsstr)
                        tfedge.rule = rule
                        node.tfedges.append(tfedge)

            if len(node.tfedges) == 0:  # no translation hyperedge
                # add a default translation hyperedge
                deflhs = "%s(%s)" % (node.label, " ".join(sub.label for sub in edge.subs))
                defrhs = ["x%d" % i for i, _ in enumerate(edge.subs)] # N.B.: do not supply str
                defrule = Rule(deflhs, defrhs, self.deffields)
                tfedge = Hyperedge(node, edge.subs,\
                                   Vector(self.deffields), edge.subs)
                tfedge.rule = defrule
                ruleset.add_rule(defrule)
                node.tfedges.append(tfedge)
 
