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

from forest import Forest
from bleu import Bleu

import gflags as flags
FLAGS=flags.FLAGS

from model import Model

def inside_outside(forest):

    forest.bestparse(weights) ## inside
    
    forest.root.alpha = 0

    for node in forest.reverse():   ## top-down
        if not hasattr(node, "alpha"):
            node.unreachable = True
        else:
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
            if hasattr(sub, "unreachable") or sub.merit > threshold:
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
        if not hasattr(node, "unreachable") and node.merit <= threshold:  ## node pruning
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

#     optparser.add_option("-s", "--suffix", dest="suffix", help="dump suffix (1.suffix)", metavar="SUF")
#     optparser.add_option("-S", "--start", dest="startid", help="dump start id", \
#                          metavar="ID", type=int, default=1)

    flags.DEFINE_float("prob", None, "score threshold", short_name="p")
    flags.DEFINE_boolean("oracle", False, "compute oracle after pruning")
    flags.DEFINE_boolean("out", True, "output pruned forest (to stdout)")
    flags.DEFINE_string("suffix", None, "suffix for dumping (1.<suffix>)", short_name="s")
    flags.DEFINE_integer("startid", 1, "start id for dumping")

    from ngram import Ngram # defines --lm and --order    

    argv = FLAGS(sys.argv)

    if FLAGS.prob is None:
        print >> logs, "Error: must specify pruning threshold by -p" + str(FLAGS)
        sys.exit(1)

    weights = Model.cmdline_model()
    lm = None
    if FLAGS.lm:
        lm = Ngram.cmdline_ngram()
        weights["lm"] *= FLAGS.lmratio    
    
    myoraclebleus = Bleu()
    myscores = 0

    total_nodes = total_edges = old_nodes = old_edges = 0
    
    for i, forest in enumerate(Forest.load("-", lm=lm)):
        if forest is None:
            print
            continue
        
        if FLAGS.prob > 0: # TODO: 1-best in case gap = 0
            prune(forest, FLAGS.prob)

        if FLAGS.oracle: #new
            bleu, hyp, fv, edgelist = forest.compute_oracle(weights, 0, 1, store_oracle=True)
            ##print >> logs, forest.root.oracle_edge
            bleu = forest.bleu.rescore(hyp)
            mscore = weights.dot(fv)
            print  >> logs, "moracle\tscore=%.4lf\tbleu+1=%.4lf\tlenratio=%.2lf\n%s" % \
                  (mscore, forest.bleu.fscore(), forest.bleu.ratio(), hyp)
            
            myoraclebleus += forest.bleu.copy()
            myscores += mscore

        if FLAGS.out:
            if FLAGS.suffix is not None:
                forest.dump(open("%d.%s" % (i+FLAGS.startid, FLAGS.suffix), "wt"))
            else:
                forest.dump()

        if i % 10 == 9:
                print >> logs, "%d forests pruned, avg new size: %.1lf %.1lf (survival ratio: %4.1lf%% %4.1lf%%)" % \
                            (i+1, total_nodes / (i+1.), total_edges / (i+1.), \
                             total_nodes * 100. / old_nodes, total_edges * 100. / old_edges)
                
    if FLAGS.oracle:
        print >> logs,  "overall my    oracle bleu = %.4lf (%.2lf) score = %.4lf" \
              % (myoraclebleus.score_ratio() + (myscores/(i+1),))


