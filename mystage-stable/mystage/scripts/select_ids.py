#!/usr/bin/env python

''' select (a subset of) features by ids

cat bigfile | select_ids.py smallfile > bigfile.small
bigfile example:
  
0       NLogP 0
1       Rule:0:0:0:0:0:0:0:1 (RB DT _ ADJP)
2       Rule:0:0:0:0:0:0:0:1 (ADVP WP _ WHNP)
...

smallfile example:

0 -0.433559547
73 0.328667326
113 0.021336591
...
'''

import sys

smallfile = open(sys.argv[1])

def readsid():
	s = smallfile.readline()
	if not s:
		return None
	return int(s.split()[0])

sid = readsid()

for line in sys.stdin:
	fid = int(line.strip().split()[0])
	if fid == sid:
		print line,
	if fid >= sid:
		# read in next
		sid = readsid()
		if sid == None:
			break
	
		
	
	
	




