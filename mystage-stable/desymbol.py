#!/usr/bin/env python

import sys
import utility

if __name__ == "__main__":

    for line in sys.stdin:
        print utility.desymbol(line.strip())
