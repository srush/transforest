#!/usr/bin/env python

''' extract features from k-best trees from stdin (in EC format) and output feature counts file,
which will be used by a weights learner.
'''

import sys

logs = sys.stderr

from tree import Tree
import features
#import heads 

if __name__ == "__main__":
	try:
		import psyco
		psyco.full()
	except:
		pass

	maxk = 100000  #inf
	if len(sys.argv) > 1 and sys.argv[1].find("-k") >= 0:
		maxk = int(sys.argv[1][2:])
		del sys.argv[1]

	if len(sys.argv) > 1 and sys.argv[1].find("-c") >= 0:
		features.cross_lines = True
		del sys.argv[1]

	if len(sys.argv) > 1 and sys.argv[1].find("-f") >= 0:
		features.print_features = True
		del sys.argv[1]
		
	features.prep_features(sys.argv[1:], prep_weights=False)

	f = sys.stdin
	while True:
		line = f.readline()
		if line == '':
			break
		if line == '\n':
			continue
		try:
			k, tag = line.strip().split("\t")
		except:
			break  ## can finish earlier
 		print k, tag
		k = int(k)
		best_w = None
		for j in xrange(k):
			logprob = float(f.readline().strip())
			parse = f.readline().strip()
			tree = Tree.parse(parse)
			##print tree

			if j < maxk:
				fvector = features.extract(tree, tree.get_sent())
				print features.pp_fv(fvector, j)
				
				
