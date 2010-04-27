#!/usr/bin/env python

import sys
from utility import num_words, words_from_line

## cat trees.txt | filter_by_length.py [-w] [<max_len>]

if __name__ == "__main__":

	print_words = False
	if sys.argv[1] == "-w":
		## words
		print_words = True
		del sys.argv[1]

	try:
		max_len = int(sys.argv[1])
	except:
		max_len = 400

	for line in sys.stdin:

		words = words_from_line(line)
		length = len(words)
		if length <= max_len:
			print " ".join(words) if print_words else line.strip() 
	
