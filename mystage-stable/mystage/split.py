
#######################################3
#######################################3
#######################################3
#######################################3
#######################################3
#######################################3
#######################################3
#######################################3
#######################################3
#######################################3
#######################################3
#######################################3
#######################################3
#######################################3
#######################################3
#######################################3
#######################################3
#######################################3


def split_forest(forest):
	''' recursive split '''

	annotations = []
	for node in forest:
		node.parentlabel = None
		node.rootcontext = None
		node.conjcontext = None
		node.synhead = None
		
	for node in forest.reverse():
		for edge in node.edges:
			for sub in edge.subs:

				if "parentlabel" in annotations:
					nodelabel = node.label if not node.is_spurious() else node.parentlabel
					if sub.parentlabel is not None and sub.parentlabel != nodelabel:
						print >> logs, edge, "==", sub, "oldlabel=", sub.parentlabel, "newlabel=", nodelabel
					sub.parentlabel = nodelabel

				if "rootcontext" in annotations:
					pass
				
				if "conjcontext" in annotations:
					pass
				

	htype = heads.SYN
	for node in forest:
		if node.is_terminal():
			node.synhead = heads.LexHead(node)
		else:
			if node.is_spurious():
				for edge in node.edges:
					newhead = edge.subs[0].synhead
					if node.synhead is not None and node.synhead.word != newhead.word:
						print >> logs, edge, "   oldhead=", node.synhead, "newlabel=", newhead
					node.synhead = newhead				
			else:
				for edge in node.edges:
					node.subs = edge.subs				
					newhead = heads.headchild(htype, node).synhead
					if node.synhead is not None and node.synhead.word != newhead.word:
						print >> logs, edge, "   oldhead=", node.synhead, "newlabel=", newhead
					node.synhead = newhead
					
	
