#!/usr/bin/env python

import sys
import math
import os
import time

from utility import words_from_line

logs = sys.stderr

def run(dirname, cmd, bg=False):

	realcmd = "cd " + dirname + "; " + cmd
	if bg:
		realcmd += " &"
	print >> logs, "running command:\n  " + realcmd
	ret = os.system(realcmd)

	if ret != 0:
		print >> logs, "FAILED! exit %d --------------" % ret
	
if __name__ == "__main__":

	import optparse
	optparser = optparse.OptionParser(usage="usage: cat <goldtrees> | %prog [options (-h for details)]")
	optparser.add_option("-b", "--bg", dest="background", type=int, metavar="NUM", \
						 help="the first of every NUM folds will run in bg", default=None)
	optparser.add_option("-N", "--numfolds", dest="numfolds", type=int, \
						 help="number of folds", default=20, metavar="NUM")
	optparser.add_option("-d", "--dev", dest="devfile", type=str, \
						 help="dev file", default="22.cleangold", metavar="FILE")
	optparser.add_option("-o", "--output", dest="outdir", type=str, \
						 help="output to dir", default="folds", metavar="DIR")
	optparser.add_option("-f", "--from", dest="fromstep", type=int, \
						 help="start from step (0:folds, 1: train, 2:parse)", metavar="STEP", default=0)
	optparser.add_option("-F", "--fold", dest="onlyfold", type=int, \
						 help="only works on fold FOLD", metavar="FOLD", default=None)
	optparser.add_option("-k", "--kbest", dest="k", metavar="K", default=50,
						 help="nbest list size (default: 50)")

	(opts, args) = optparser.parse_args()
	
	goldtrees = sys.stdin.readlines()
	totalnum = len(goldtrees)	
	foldlen = int (math.ceil (float(totalnum) / opts.numfolds))  ## last fold is less than others

	thisdir = {}
	for i in range(opts.numfolds) + [100]:
		thisdir[i] = "%s/%d" % (opts.outdir, i)
	

######### MAKING FOLDS *********************************************

	print >> logs, "**************************** makeing folds ************************"
	if opts.fromstep <= 0:
		print >> logs, "%d lines in total, %d folds, each with %d lines (except for the last fold)" \
			  % (totalnum, opts.numfolds, foldlen)
		print >> logs, "output to %s/" % opts.outdir

##		os.system("mkdir " + opts.outdir)
		for i in xrange(opts.numfolds):
			if opts.onlyfold is not None and i != opts.onlyfold:
				continue
			left = i*foldlen
			right = min(totalnum, (i+1)*foldlen)

			print >> logs, i, "\t [%d, %d) \t%d lines" % (left, right, right-left)

			os.system("mkdir " + thisdir[i])

			infold_input = open(thisdir[i] + "/toparse.ecinput", "wt")
			outfold_gold = open(thisdir[i] + "/totrain.cleangold", "wt")
			infold_gold = open(thisdir[i] + "/toparse.cleangold", "wt")

			for j, line in enumerate(goldtrees[left : right]):
				start = "<s small.%d.%d>" % (i, j+1)
				print >> infold_input,  start, " ".join(words_from_line(line)), "</s>"

			print >> infold_gold, "".join(goldtrees[left : right]),

			print >> outfold_gold, "".join(goldtrees[:left] + goldtrees[right:]),
		
####### TRAINING *****************

	print >> logs, "**************************** training folds ************************"
	if opts.fromstep <= 1:

		traindir = os.environ["HOME"] + "/rerank/first-stage/TRAIN"
		trainscript = traindir + "/allScript"

		for i in xrange(opts.numfolds):

			if opts.onlyfold is not None and i != opts.onlyfold:
				continue

			bg = opts.background is not None and (i % opts.background == 0)

			starttime = time.time()
			
			dirs = {'traindir' : traindir, 'trainscript' : trainscript, 
					'totrain' : 'totrain.cleangold', 'dev' : '../../' + opts.devfile}
			print >> logs, "training on fold %d" % i
			run(thisdir[i], "cp -pr %(traindir)s/../DATA/EN ." % dirs) ## copy basic data
			run(thisdir[i], "%(trainscript)s  EN  %(totrain)s  %(dev)s >train.log 2>train.log2" % dirs, bg)

			print >> logs, "training on fold %d done in %.2lf secs." % (i, time.time() - starttime)
			
		
####### PARSING / FORESTING ******

	print >> logs, "**************************** parsing folds ************************"
 
	if opts.fromstep <= 2:

		parser = os.environ["HOME"] + "/rerank/first-stage/PARSE/parseIt"

		for i in xrange(opts.numfolds):
			if opts.onlyfold is not None and i != opts.onlyfold:
				continue

			bg = opts.background is not None and (i % opts.background == 0)
			
			options = "-t1 -K -l399 -N50 -f1" ## 1.forest, 2.forest, etc.

			starttime = time.time()
			
			dirs = {'parser' : parser, 'toparse' : 'toparse.ecinput', 'options' : options}
			print >> logs, "parsing on fold %d" % i

			run(thisdir[i], "rm -rf *.forest")
			run(thisdir[i], "%(parser)s  %(options)s  EN/  %(toparse)s" % dirs + \
				" | sed -e 's/^(S1 /(TOP /g' > %dbest" % opts.k + \  
				"; rename 's/^/small.%d./g' *.forest" % i, bg)
			print >> logs

			print >> logs, "parsing on fold %d done in %.2lf secs." % (i, time.time() - starttime)
			
		
