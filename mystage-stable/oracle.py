#!/usr/bin/env python

''' Forest-oracles (PARSEVAL F_1) using Dynamic Programming.
    punc deletion is still buggy and can\'t find an easy workout.

    current workout:
    1. in forest oracle, do not delete puncs.
    2. then use parseval.py to compute REAL parseval. (report two sets of numbers)
'''

import sys, copy, time

logs = sys.stderr

from forest import Forest
from readkbest import readonebest
from parseval import Parseval, merge_label, merge_labels
from tree import Tree
from prune import prune
from utility import is_punc, desymbol
from node_and_hyperedge import Hyperedge

debug = False

class RES(tuple):
    ''' (num_matched_edges, score, tree_str, edgelist)
        this is used by the Oracle class, which is a mapping from num_test_edges to RES.
    '''

    def __add__(self, other):
        ''' point-wise addition '''
##        return RES((self[0] + other[0], self[1] + other[1], self[2] + " " + other[2], self[3] + other[3])) 
        return RES((self[0] + other[0], self[1] + other[1], self[2] + other[2])) 

    def __mul__(self, other):
        ''' point-wise addition, with assembling :) '''
##        res = "(%s %s)" % (other[2], self[2].strip()) if other[2] != "" else self[2].strip()
##        return RES((self[0] + other[0], self[1] + other[1], res, other[3] + self[3]))

        ## reversed, so that TOP will appear first
        return RES((self[0] + other[0], self[1] + other[1], other[2] + self[2]))  

    def __str__(self):
##        return "(%d, %.2lf, %s)" % self[:3] + "\n\t" + "\n\t".join(map(str, self[3])) + "\n"
        return "(%d, %.2lf)" % self[:2] + "\n\t" + "\n\t".join(map(str, self[2])) + "\n"

    @staticmethod
    def unit(val=None):
##        tree_str = "" if val is None else val
##        return RES((0, 0, tree_str, []))
        return RES((0, 0, []))
        
class Oracles(dict):
    ''' a mapping from num_test_edges to max_num_matched_edges (actually RES tuple).
        provides convolution *, pointwise +=, and scaling *, like a LM-integration.    
    '''

#     def __init__(self, d={}):
#         dict.__init__(self, d)
        ## careful: this only works if d is a singleton, which is the case now
#         self.reverse = dict([(b[0], a) for (a, b) in d.items()])

#     def __setitem(self, key, value):
#         v = value[0]
#         if key <= self.reverse.get(v, 100000):
#             dict.__setitem__(self, key, value)
#             self.reverse[v] = key 
        
    def __mul__(self, other):
        ''' convolution '''

        new = Oracles()
        for a in self:
            for b in other:
                newvalue = self[a] + other[b]
                if a+b not in new or (newvalue > new[a+b]):
                    new[a+b] = newvalue
        return new
                
    def __iadd__(self, other):
        ''' pointwise addition (the add operator is "max"). caution: += has to return self '''
        for num in self:
            if num in other and (other[num] > self[num]):
                self[num] = other[num]
        for num in other:
            if num not in self:
                self[num] = other[num]

        return self

    def __rmul__(self, (key, value)):  ## scale (or shift)
        new = Oracles()
        for num in self:
            new[num + key] = self[num] * value   ## calls RES.__mul__ to do tree assembling
        return new

    def __str__(self):
        return "  " + "  ".join(["%d: %s" % (x, self[x]) for x in self])

    @staticmethod
    def unit(val=None):
        ''' unit-element for * '''

        return Oracles({0: RES.unit(val)})

def check_puncs(forest, pos2):
    '''returns idx_mapping and the modified forest.
    CAUTION: can not modify the original forest.
    CAUTION: delete redundant layers'''

    newforest = forest.copy()
    assert len(newforest) == len(pos2), "different sentence lengths!\n%s\n%s" % (str(test_tree), str(gold_tree))

    idx_mapping = {}
    j = 0
    last_is_punc = True
    for i, a in enumerate(pos2):
        if not last_is_punc:
            j += 1
        idx_mapping [i] = j
        
        # delete the non-consistent tags of this word in the newforest
        for node in newforest.cells[(i, i+1)]:
            if node.is_terminal() or node.sp_terminal():
                node.deleted = is_punc(node.label) ^ is_punc(a)
                if node.deleted:
                    print >> logs, node, "deleted"
                    
        last_is_punc = is_punc(a)

    if not last_is_punc:
        j += 1
    idx_mapping[i+1] = j

    # TODO: CLEAN UP THIS PART!
    newforest.nodeorder = [node for node in newforest if not node.is_terminal() or not node.deleted]
    newforest.nodes = {}
    for node in newforest:
        newforest.nodes[node.iden] = node
    newforest.rehash()
    for node in newforest:
        if not node.is_terminal():
            mapped_span = node.mapped_span(idx_mapping)
            newedges = []
            for edge in node.edges:
                if edge.unary_cycle():
                    print >> logs, edge, "deleted (cycle)"
                else:
                    for sub in edge.subs:
                        if sub.is_terminal() and sub.deleted:
                            print >> logs, edge, "deleted (punc)"
                            break
                        if not node.is_root() and sub.label == node.label \
                               and sub.mapped_span(idx_mapping) == mapped_span:
                        ## make sure no induced unary cycle
                            print >> logs, edge, "deleted (induced cycle)"
                            break                            
                    else:                        
                        newedges.append(edge)
                        
            node.edges = newedges

    return lambda x:idx_mapping[x], newforest

def prune(results):
    '''prune away obviously worse items.
       test: match
       if m1 <= m2 and t1>=t2 then remove (m1, t1);
       start from the lowest.'''

    lst = sorted(results.items())

    deleted = set()
    for x in lst:
        t1, m1 = x[0], x[1][0]
        for j, y in enumerate(lst):
            if y is not x and j not in deleted:
                t2, m2 = y[0], y[1][0]            
                if m2 <= m1 and t2 >= t1:
                    ## y is deleted
                    deleted.add(j)
                    del results[t2]
        

def get_edgelist(node):

    assert hasattr(node, "oracle_edge"), node
    edge = node.oracle_edge

    a = [edge]
    for sub in edge.subs:
        if not sub.is_terminal():
            a += get_edgelist(sub)
    return a    

implicit_oracle = False

def extract_oracle(forest):
    '''oracle is already stored implicitly in the forest
       returns best_score, best_parseval, best_tree, edgelist
    '''
    global implicit_oracle
    implicit_oracle = True
    edgelist = get_edgelist(forest.root)
    fv = Hyperedge.deriv2fvector(edgelist)
    tr = Hyperedge.deriv2tree(edgelist)
    return fv[0], Parseval(), tr, edgelist
    
    
def forest_oracle(forest, goldtree, del_puncs=False, prune_results=False):
    ''' returns best_score, best_parseval, best_tree, edgelist
           now non-recursive topol-sort-style
    '''

    if hasattr(forest.root, "oracle_edge"):
        return extract_oracle(forest)
    
    ## modifies forest also!!
    if del_puncs:
        idx_mapping, newforest = check_puncs(forest, goldtree.tag_seq)
    else:
        idx_mapping, newforest = lambda x:x, forest

    goldspans = merge_labels(goldtree.all_label_spans(), idx_mapping)
    goldbrs = set(goldspans) ## including TOP

    for node in newforest:
        if node.is_terminal():
            results = Oracles.unit("(%s %s)" % (node.label, node.word))  ## multiplication unit
            
        else:
            a, b = (0, 0) if node.is_spurious() \
                   else ((1, 1) if (merge_label((node.label, node.span), idx_mapping) in goldbrs) \
                         else (1, 0))

            label = "" if node.is_spurious() else node.label
            results = Oracles()     ## addition unit
            for edge in node.edges:
                edgeres = Oracles.unit()  ## multiplication unit

                for sub in edge.subs:
                    assert hasattr(sub, "oracles"), "%s ; %s ; %s" % (node, sub, edge)
                    edgeres = edgeres * sub.oracles

##                nodehead = (a, RES((b, -edge.fvector[0], label, [edge])))   ## originally there is label
                assert 0 in edge.fvector, edge
                nodehead = (a, RES((b, -edge.fvector[0], [edge])))   
                results += nodehead * edgeres   ## mul

        if prune_results:
            prune(results)
        node.oracles = results
        if debug:
            print >> logs, node.labelspan(), "\n", results, "----------"

    res = (-1, RES((-1, 0, []))) * newforest.root.oracles   ## scale, remove TOP match

    num_gold = len(goldspans) - 1 ## omit TOP.  N.B. goldspans, not brackets! (NP (NP ...))

    best_parseval = None
    for num_test in res:
##        num_matched, score, tree_str, edgelist = res[num_test]
        num_matched, score, edgelist = res[num_test]
        this = Parseval.get_parseval(num_matched, num_test, num_gold)
        if best_parseval is None or this < best_parseval:
            best_parseval = this
            best_score = score
##            best_tree = tree_str
            best_edgelist = edgelist

    best_tree = Hyperedge.deriv2tree(best_edgelist)

    ## annotate the forest for oracle so that next-time you can preload oracle
    for edge in best_edgelist:
        edge.head.oracle_edge = edge
    
    ## very careful here: desymbol !
##    return -best_score, best_parseval, Tree.parse(desymbol(best_tree)), best_edgelist
    return -best_score, best_parseval, best_tree, best_edgelist

if __name__ == "__main__":

    import optparse
    optparser = optparse.OptionParser(usage="usage: cat <forests> | %prog [options (-h for details)]")
    optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", \
                         help="output sentence by sentence parseval", default=False)
    optparser.add_option("-d", "--debug", dest="debug", action="store_true", \
                         help="debug info within each sentence", default=False)
    optparser.add_option("-g", "--gold", dest="goldfile", \
                         help="gold file", metavar="FILE", default=None)
    optparser.add_option("-R", "--range", dest="range", \
                         help="test pruning (e.g., 5:15:25)", metavar="RANGE", default=None)
    optparser.add_option("-s", "--suffix", dest="suffix", metavar="SUF", default=None,
                         help="dump the pruned forests as $i.SUF$p (must use with -R)")
    optparser.add_option("-p", "--prune", dest="prune_results", action="store_true", default=False,
                         help="prune obviously worse items on-the-fly")

    (opts, args) = optparser.parse_args()
    
    debug = opts.debug
    if opts.goldfile is not None:
        goldtrees = readonebest(opts.goldfile)
    

    prange = None
    if opts.range is not None:
        prange = eval("[%s]" % opts.range.replace(":", ","))
        prange.sort(reverse=True)

        pruned_parseval = dict([(p, Parseval()) for p in prange])
        pruned_real_parseval = dict([(p, Parseval()) for p in prange])

    all_parseval = Parseval()
    all_real_parseval = Parseval()
    onebest_parseval = Parseval()

    start_time = time.time()
    total_time = 0
    for i, f in enumerate(Forest.load("-")):

        bres = f.bestparse(adjust=False) ## no adjustment
        base_score = bres[0]

        if opts.goldfile is not None:
            f.goldtree = goldtrees.next()

##        start = time.time()
        best_score, best_parseval, best_tree, edgelist = \
                    forest_oracle(f, f.goldtree, prune_results=opts.prune_results)
##        this_time = time.time() - start
        
##        print >> logs, "forest-oracle computed in %.2lf secs" % (this_time)
##        total_time += this_time

        if opts.verbose:
    
            print best_score, "\t", best_tree
            if opts.debug:
                print "\t" + "\n\t".join(map(str, edgelist))
            print i+1, "  %d w %d n %d e\t" % ((len(f.sent),) + f.size()),\
                  best_parseval, "%.4lf" % (best_score - base_score)

        
        onebest_parseval += Parseval(bres[1], f.goldtree)
        real_parseval = Parseval(best_tree, f.goldtree)
        all_real_parseval += real_parseval
        ##assert real_parseval == best_parseval
        ## N.B.: can't make this comparison work, so keep it separate.

        all_parseval += best_parseval

        if prange is None: 
            ## dump oracle-annotated forest
            if opts.suffix is not None:
                f.dump("%d.%s" % (i+1, opts.suffix))            

        else:
            for p in prange:

                prune(f, p)
                sc, parseval, tr = forest_oracle(f, f.goldtree)
                pruned_parseval[p] += parseval
                pruned_real_parseval[p] += Parseval(tr, f.goldtree)

                if opts.suffix is not None:
                    f.dump("%d.%s%d" % (i+1, opts.suffix, p))
                

    print "1-best (real)", onebest_parseval
    if not implicit_oracle:
        print "forest (punc)", all_parseval
    print "forest (real)", all_real_parseval

    total_time = time.time() - start_time
    print >> logs, "%d forests oracles computed in %.2lf secs (avg %.2lf secs per sent)" % \
          (i+1, total_time, total_time/(i+1))

    if prange is not None:
        for p in prange:
            print "p %4.1lf (punc)" % p, pruned_parseval[p]
            print "p %4.1lf (real)" % p, pruned_real_parseval[p]

    
