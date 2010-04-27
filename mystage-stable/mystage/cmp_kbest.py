#!/usr/bin/env python

import sys, math

## compare k-best list for scores (and k-best length) only; trees are omitted in the preprocessing.
## note that there are some precision error b/w charniak and the forest, as the result of
## breaking down of edge scores. however, if the lengths of the k-best lists are different, it is
## considered a serious problem -- often due to duplicate-removals. 
##
## usage:
##    cat tmp | grep "^[-]*\d+" >tmp.scores
##    cat 50best | grep "^[-]*\d+" >tmp.ec-scores
##    cmp_kbest.py tmp.scores tmp.ec-scores

## 50     1
## -123.4352
## -124.3564
## ...

precision = 0.00021

if __name__ == "__main__":

	filea = open(sys.argv[1])
	fileb = open(sys.argv[2])
	
	for linea, lineb in zip(filea, fileb):
		if linea.find("\t") >=0:
			assert linea == lineb, "%s\t\t%s" % (linea.strip(), lineb.strip())
			
		else:
			sa = float(linea.strip())
			sb = float(lineb.strip())
			
			if math.fabs(sa - sb) > precision:
				print "%.4lf\t\t%.4lf" % (sa, sb)
