#!/usr/bin/env python

''' A Linked List (for Feature Vector)
'''

import sys
import copy
import math

from utility import getfile

logs = sys.stderr

class LinkedList(object):

	__slots__ = "start", "end", "_size"

	def __len__(self):
		return self._size

	def __iadd__(self, other):
		assert isinstance(other, LinkedList), "can only add LinkedList with LinkedList"
		self.end = other.start
		self.end = other.end
		self._size += other._size

	
			
		
