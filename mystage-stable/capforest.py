#!/usr/bin/env python

''' capitalize old forests by comparing to ec input.
      cat 23.forest | capforest.py 23.ecinput > 23.capforest
      
    ecinput is of format:

    <s big.sec23.1> No , it was n\'t Black Monday . </s>

'''

import sys, re

logs = sys.stderr

extract = re.compile(r'<s (.*?)> (.*?) </s>')

if __name__ == "__main__":

    def getnext():
        ecinput = open(sys.argv[1])
        for ecline in ecinput:
            if ecline[0]=="<":
                ret = extract.match(ecline).groups()[1]
            else:
                ret = ecline[:-1]
            yield ret

    tags = getnext()
        
    sent = tags.next()
    # now tag should be "wsj_2300.1" and sent should be the (capitalized) sent.
        
    for line in sys.stdin:
        line = line[:-1]  # DO NOT strip: keeping initial tabs
        if line != "" and line[0]!="\t" and not line[0].isdigit() and line[0]!="(":
            ## assumes that tags are alphanumeric
            print "%s\t%s" % (line.split("\t")[0], sent)

            try:
                sent = tags.next()
            except:
                pass ## last one
        else:
            print line
