#!/usr/bin/env python

import sys, math

logs = sys.stderr

from forest import Forest

top_down = True
precision = 1e-4

def dir_edge(head, tail, features=""):

	xx = (head, tail) if not top_down else (tail, head)

	return "%s -> %s [%s];" % (xx[0], xx[1], features)

def dump2dot(forest):

	'''output to stdout'''
	# wsj_00.00       No , it was n't Black Monday .
	# 199
	# 1	DT [0-1]	0 ||| 12321=... 46456=...
	# ...
	# 6	NP [0-2]	1 ||| 21213=... 7987=...
	# 	1 4 ||| 0=-5.342
	# ...

	print "digraph Forest {"  ##, "_".join(forest.sent))

##	print "graph [ranksep=1]"

	colors = ["brown", "green", "magenta", "red", "orange", "gray"]

	terminals = [[] for i in range(forest.len+1)]
	cells = {}	

	bestmerit = f.root.merit
	
	for i, node in enumerate(forest):
		word = "\\n %s" % (node.wrd_seq[0] if (node.span_width() == 1) \
						   else "%s .. %s" % (node.wrd_seq[0], node.wrd_seq[-1]))
		node.dot_label = "n%s" % node.iden
		node.dot_words = word
		
		if node.is_terminal():
			print "%s [label=\"%s%s\" color=%s style=filled]" % \
				  (node.dot_label, node.labelspan(separator=":"), \
				  node.dot_words, colors[node.span[0] % len(colors)])
			
			print dir_edge("start", node.dot_label, "color=\"black\" weight=100")
			terminals[node.span[0]].append(node)

		else:
			print "%s [label=\"%s%s\" %s %s]" % \
				  (node.dot_label, node.labelspan(separator=":"), node.dot_words, \
				   "shape=box color=blue" if node.is_spurious() else "", \
				   "color=black style=bold width=3 weight=200" \
				   if math.fabs(node.merit - bestmerit) < precision else "color=red")
			
			labelspan = node.labelspan(include_id=False)
			if labelspan not in cells:
				cells[labelspan] = [node]
			else:
				cells[labelspan].append(node)

	for i in range(forest.len):
		for node in terminals[i]:
			for nodeb in terminals[i+1]:	
				print dir_edge(nodeb.dot_label, node.dot_label, "color=\"brown\" weight=100 minlen=0")

	for span, cell in cells.items():
		print "{ rank = same; %s };" % " ".join([node.dot_label for node in cell])
		for i in range(len(cell)-1):
			nodea, nodeb = cell[i], cell[i+1]
			print dir_edge(nodea.dot_label, nodeb.dot_label, "color=\"blue\" weight=2000 minlen=0 style=bold dir=both")
			

	counter = 0
	for node in forest:
		for i, edge in enumerate(node.edges):
			## make a dummy AND-node

			counter += 1			
			if math.fabs(edge.merit - bestmerit) < precision:
				col = "black"
				meritstyle = "style=bold width=2 weight=200"
			else:
				col = colors[counter % len(colors)]
				meritstyle = ""
				
			meritstyle += " color=%s" % col
				
			edge.dot_label = "e%s_%d" % (node.iden, i)

			## dummy AND-node
			print "%s [color=%s label=\"\" width=0 height=0];" % (edge.dot_label, col)

			print dir_edge(edge.dot_label, node.dot_label, \
						   "label=\"%.2lf\" dir=back %s" % (edge.edge_score, meritstyle))

			for sub in edge.subs:
				print dir_edge(sub.dot_label, edge.dot_label, "arrowhead=none " + meritstyle)

	print "}"

if __name__ == "__main__":

	import optparse
	optparser = optparse.OptionParser(usage="usage: cat <forest> | %prog [options (-h for details)]")

	f = Forest.load("-").next()

	print >> logs, f.bestparse()[1]
	
	from prune import *
	inside_outside(f)
	dump2dot(f)
