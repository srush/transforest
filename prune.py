#!/usr/bin/env python

''' relatively-useless pruning, after Jon Graehl.
    also called posterior pruning, used by Charniak and Johnson 2005, but only on nodes.

    basically, it is an inside-outside algorithm, with (+, max) semiring.

    \beta (leaf) = 0    
    \beta (n) = max_{e \in BS(n)} \beta(e)
    \beta (e) = max_{n_i \in tails(e)} \beta(n_i) + w(e)

    (bottom-up of \beta is done in node_and_hyperedge.py: Node.bestparse())

    \alpha (root) = 0
    \merit (e) = \alpha (n) + \beta (e),   n = head(e)     ## alpha-beta
    \alpha (n_i) max= \merit (e) - \beta (n_i),  for n_i in tails(e)

    \merit (n) = \alpha (n) + \beta (n) = max_{e \in BS(n)} \alphabeta (e)
'''

import sys, time

logs = sys.stderr

from forest import Forest, get_weights
from bleu import Bleu

def inside_outside(forest):

    forest.bestparse(weights) ## inside
    
    forest.root.alpha = 0

    for node in forest.reverse():   ## top-down
        assert hasattr(node, "alpha"), node
        node.merit = node.alpha + node.beta
    
        for edge in node.edges:
            edge.merit = node.alpha + edge.beta
            for sub in edge.subs:
                score = edge.merit - sub.beta
                if not hasattr(sub, "alpha") or score < sub.alpha:
                    sub.alpha = score
    
def prune(forest, gap, delete=True, do_merits=True):
    ''' Known issue: not 100% correct w.r.t. floating point errors.'''

    def check_subs(edge, threshold):
        ''' check if every tail falls in the beam. '''
        for sub in edge.subs:
            if sub.merit > threshold:
                return False
        return True

    start_time = time.time()

    if do_merits:
        inside_outside(forest)

    oldsize = forest.size()
    
    threshold = forest.root.merit + gap    
    newnodes = {}
    neworder = []

    kleinedges = 0
    for node in forest:
        iden = node.iden
        if node.merit <= threshold:  ## node pruning
            newnodes[iden] = node
            neworder.append(node)
            node.edges = [e for e in node.edges if (e.merit <= threshold and check_subs(e, threshold))] 

        else:
            kleinedges += len(node.edges)
            del node

    del forest.nodes
    del forest.nodeorder
    
    forest.nodes = newnodes
    forest.nodeorder = neworder

    forest.rehash() ## important update for various statistics
    
    newsize = forest.size()
    
    print >> logs, "%s gap %4.1lf, %4d nodes, %5d edges remained. prune ratio = %4.1lf%%, %4.1lf%% (%4.1lf%%)" \
          % (forest.tag, gap, newsize[0], newsize[1], \
             (oldsize[0] - newsize[0])*100.0 / oldsize[0], (oldsize[1] - newsize[1])*100.0 / oldsize[1],\
             kleinedges*100.0/oldsize[1]),
    print >> logs, "done in %.2lf secs" % (time.time() - start_time)

    global total_nodes, total_edges, old_nodes, old_edges
    total_nodes += newsize[0]
    total_edges += newsize[1]
    old_nodes += oldsize[0]
    old_edges += oldsize[1]
    

if __name__ == "__main__":

    import optparse
    optparser = optparse.OptionParser(usage="usage: cat <forest> | %prog -p <GAP> [options (-h for details)]")
    optparser.add_option("-p", "--prob", dest="gap", type=float, help="merit threshold", metavar="PROB_GAP")
    optparser.add_option("-s", "--suffix", dest="suffix", help="dump suffix (1.suffix)", metavar="SUF")
    optparser.add_option("-S", "--start", dest="startid", help="dump start id", \
                         metavar="ID", type=int, default=1)
    optparser.add_option("-q", "--quiet", dest="quiet", action="store_true", help="no dumping", default=False)
    optparser.add_option("-O", "--oracle", dest="oracle", action="store_true", help="oracle", default=False)
    ## weights
    optparser.add_option("-w", "--weights", dest="weights", type=str, help="weights file or str", metavar="WEIGHTS", default="lm1=2 gt_prob=1")

    (opts, args) = optparser.parse_args()

    weights = get_weights(opts.weights)

    if opts.gap is None:
        optparser.error("must specify GAP.")

##    if opts.oracle:
        
##         from oracle import forest_oracle
##         from parseval import Parseval
        
##        realpp = Parseval()

    myoraclebleus = Bleu()
    myscores = 0

    total_nodes = total_edges = old_nodes = old_edges = 0
    
    for i, forest in enumerate(Forest.load("-")):
        if forest is None:
            print
            continue
        
        if opts.gap > 0: # TODO: 1-best in case gap = 0
            prune(forest, opts.gap)

        if opts.oracle: #new
            bleu, hyp, fv, edgelist = forest.compute_oracle(weights, 0, 1, store_oracle=True)
            ##print >> logs, forest.root.oracle_edge
            bleu = forest.bleu.rescore(hyp)
            mscore = weights.dot(fv)
            print  >> logs, "moracle\tscore=%.4lf\tbleu+1=%.4lf\tlenratio=%.2lf\n%s" % \
                  (mscore, forest.bleu.fscore(), forest.bleu.ratio(), hyp)
            
            myoraclebleus += forest.bleu.copy()
            myscores += mscore

##             sc, parseval, tr, edgelist = forest_oracle(forest, forest.goldtree)
##             realpp += Parseval(tr, forest.goldtree)
##             for edge in edgelist:
##                 edge.head.oracle_edge = edge
##                 edge.is_oracle = True

        #forest.tag = "sent.%d" % (i+opts.startid)
        if not opts.quiet:
            if opts.suffix is not None:
                forest.dump(open("%d.%s" % (i+opts.startid, opts.suffix), "wt"))
            else:
                forest.dump()

        if i % 10 == 9:
                print >> logs, "%d forests pruned, avg new size: %.1lf %.1lf (survival ratio: %4.1lf%% %4.1lf%%)" % \
                            (i+1, total_nodes / (i+1.), total_edges / (i+1.), \
                             total_nodes * 100. / old_nodes, total_edges * 100. / old_edges)
                
    if opts.oracle:
        print >> logs,  "overall my    oracle bleu = %.4lf (%.2lf) score = %.4lf" \
              % (myoraclebleus.score_ratio() + (myscores/(i+1),))


