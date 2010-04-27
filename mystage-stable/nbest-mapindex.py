#!/usr/bin/env python

''' given index file, select the parses '''

import sys

if __name__ == "__main__":

	nbestfile = sys.stdin
	indexfile = open(sys.argv[1])
	
	for indexline in indexfile:
		
		index = int(indexline.strip())

		num = int(nbestfile.readline().strip().split("\t")[0])
		for k in range(num):			
			p = nbestfile.readline()
			if p[0]=="-":
				p = nbestfile.readline() ## skipping the prob line
			if k == index:
				print p.strip()
		assert index <= k, "out of range!"
			
			
